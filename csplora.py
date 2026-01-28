"""
C-SPLoRA: Confidence-guided Structural Planning for LoRA
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os
import json
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


@dataclass
class CSpLoRAConfig:
    """Configuration for C-SPLoRA."""
    model_id: str
    task_name: str
    target_modules: List[str]
    task_type: TaskType
    cache_dir: str

    gamma: float = 1.0
    r_base: int = 8
    r_min: int = 2
    r_max_factor: float = 4.0
    tau_scale: float = 1.0
    skip_smooth: bool = False
    skip_normalize: bool = False
    R_tot: int | None = None
    device: str = "cuda"

    probe_r: int = 2
    probe_alpha: int = 8
    probe_lr: float = 5e-4
    probe_mode: str = "full_lowmem"
    probe_use_amp: bool = True
    probe_group_size: int | None = None
    probe_checkpoint: bool = True

    adaptive_steps: bool = True
    min_probe_steps: int = 50
    max_probe_steps: int = 500
    step_convergence_threshold: float = 0.02
    step_check_interval: int = 10
    step_patience: int = 3
    step_top_k: int = 20
    step_rank_tolerance: int = 5

    adaptive_rho: bool = True
    rho_min: float = 0.6
    rho_max: float = 1.0
    rho_method: str = "coverage"
    rho_coverage_target: float = 0.95
    rho_fixed: float = 0.9

    importance_metric: str = "taylor"
    skip_layer_normalize: bool = True

    max_probe_steps_legacy: int = 200
    probe_coverage: float = 0.0
    probe_ensemble: int = 1
    probe_steps_per_group: int | None = None


class CSpLoRAPlanner:
    """C-SPLoRA Planner with adaptive step and rho selection."""

    def __init__(self, cfg: CSpLoRAConfig):
        self.cfg = cfg
        self.layer_capacity: Dict[str, int] = {}
        self.convergence_history: List[Dict[str, float]] = []
        self.adaptive_stats: Dict[str, Any] = {}

        raw_id = os.path.basename(cfg.model_id)
        safe_model_id = raw_id.replace("/", "_").replace(":", "_")

        metric_suffix = f"_{cfg.importance_metric}" if cfg.importance_metric != "taylor" else ""
        fname = f"{cfg.task_name}_ada_scores{metric_suffix}.json"
        self.cache_path = os.path.join(cfg.cache_dir, safe_model_id, fname)

    def plan(self, base_model: nn.Module, probe_dataloader) -> Tuple[Dict[str, int], Dict[str, float]]:
        scores = self._try_load_scores()

        if scores is None:
            print(f"[CSPLoRA] Running probe...")
            scores, probe_stats = self._run_adaptive_probe(base_model, probe_dataloader)
            self.adaptive_stats.update(probe_stats)
            self._save_scores(scores)
        else:
            print(f"[CSPLoRA] Loaded cached scores")

        valid_scores = {k: v for k, v in scores.items() if v > 0 and self._is_target_module(k)}
        num_potential_layers = len(valid_scores)
        if num_potential_layers == 0:
            return {}, scores

        if self.cfg.R_tot is None or self.cfg.R_tot <= 0:
            self.cfg.R_tot = num_potential_layers * self.cfg.r_base

        rho = self._compute_adaptive_rho(valid_scores) if self.cfg.adaptive_rho else self.cfg.rho_fixed
        self.adaptive_stats['final_rho'] = rho

        layer_ranks = self._allocate_ranks(valid_scores, rho)
        self._print_top_ranks(layer_ranks)
        self._print_adaptive_summary()

        return layer_ranks, scores

    def _run_adaptive_probe(self, base_model: nn.Module, probe_dataloader) -> Tuple[Dict[str, float], Dict]:
        cfg = self.cfg
        probe_model = base_model
        probe_model.to(cfg.device)
        probe_model.eval()

        prev_gc = getattr(probe_model, "is_gradient_checkpointing", False)
        if cfg.probe_checkpoint:
            self._toggle_gradient_checkpointing(probe_model, True)

        for p in probe_model.parameters():
            p.requires_grad = False

        target_params = []
        for name, param in probe_model.named_parameters():
            if "lora_" in name or "adapter" in name:
                continue
            clean_name = self._clean_layer_name(name)
            if self._is_target_module(clean_name):
                target_params.append((name, param))

        if not target_params:
            if cfg.probe_checkpoint and not prev_gc:
                self._toggle_gradient_checkpointing(probe_model, False)
            return {}, {}

        groups = self._make_groups(target_params, cfg.probe_group_size)
        amp_enabled = cfg.probe_use_amp and torch.cuda.is_available() and ("cuda" in cfg.device)

        running_g = defaultdict(float)
        step_count = 0
        converged = False
        convergence_step = -1

        energy_history: List[float] = []
        ranking_history: List[List[str]] = []
        stable_count = 0

        min_steps = cfg.min_probe_steps if cfg.adaptive_steps else cfg.max_probe_steps_legacy
        max_steps = cfg.max_probe_steps if cfg.adaptive_steps else cfg.max_probe_steps_legacy
        patience = cfg.step_patience
        top_k = cfg.step_top_k
        rank_tolerance = cfg.step_rank_tolerance

        steps_per_group = cfg.probe_steps_per_group or max_steps

        for gi, group in enumerate(groups):
            if converged:
                break

            active_names = {n for n, _ in group}
            for name, param in probe_model.named_parameters():
                param.requires_grad = name in active_names

            for step, batch in enumerate(probe_dataloader):
                if step >= steps_per_group:
                    break

                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                labels = batch.get("labels", batch["input_ids"].clone())

                probe_model.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    outputs = probe_model(**batch)
                    logits = outputs.logits

                    if logits.dim() == 2 and logits.size(-1) == 1:
                        loss = F.mse_loss(logits.squeeze(-1), labels.to(dtype=logits.dtype), reduction="mean")
                    elif logits.dim() == 2:
                        probs = logits.softmax(dim=-1)
                        label_idx = labels.to(dtype=torch.long).clamp(min=0, max=probs.size(-1) - 1)
                        conf = probs[torch.arange(logits.size(0), device=logits.device), label_idx]
                        difficulty = 1.0 - conf
                        loss_i = F.cross_entropy(logits, labels, reduction="none")
                        if cfg.gamma > 0:
                            w = torch.softmax(cfg.gamma * difficulty, dim=0)
                            loss = (w * loss_i).mean()
                        else:
                            loss = loss_i.mean()
                    else:
                        loss = outputs.loss

                loss.backward()

                for name, param in group:
                    if param.grad is None:
                        continue
                    clean_name = self._clean_layer_name(name)
                    grad = param.grad.detach()

                    metric = cfg.importance_metric.lower()
                    if metric == "fisher":
                        g_score = (grad * grad).sum().item()
                    elif metric == "taylor":
                        weight = param.data.detach()
                        g_score = (grad.abs() * weight.abs()).sum().item()
                    elif metric == "gora":
                        weight = param.data.detach()
                        g_score = (grad.abs() * weight.abs()).mean().item()
                    else:
                        g_score = (grad * grad).sum().item()

                    if math.isnan(g_score) or math.isinf(g_score):
                        continue
                    running_g[clean_name] += g_score
                    self.layer_capacity[clean_name] = self.layer_capacity.get(clean_name, 0) + param.numel()

                step_count += 1

                if cfg.adaptive_steps and step_count >= min_steps and gi == len(groups) - 1:
                    if step_count % cfg.step_check_interval == 0:
                        current_energy = sum(running_g.values()) / step_count
                        energy_history.append(current_energy)

                        energy_stable = False
                        if len(energy_history) >= 2:
                            prev_energy = energy_history[-2]
                            if prev_energy > 1e-12:
                                delta = abs(current_energy - prev_energy) / prev_energy
                            else:
                                delta = float('inf')
                            energy_stable = delta < cfg.step_convergence_threshold
                        else:
                            delta = float('inf')

                        current_raw = {k: v / step_count for k, v in running_g.items()}
                        current_capacity = {k: self.layer_capacity.get(k, 1) // max(1, step_count)
                                           for k in current_raw.keys()}

                        current_density = {}
                        for k, raw_s in current_raw.items():
                            cap = max(1, current_capacity.get(k, 1))
                            current_density[k] = raw_s / cap

                        sorted_layers = sorted(current_density.keys(),
                                              key=lambda x: current_density[x], reverse=True)
                        current_top_k = sorted_layers[:min(top_k, len(sorted_layers))]
                        ranking_history.append(current_top_k)

                        ranking_stable = False
                        max_rank_change = 0
                        if len(ranking_history) >= 2:
                            prev_top_k = ranking_history[-2]
                            ranking_stable = True
                            for i, layer in enumerate(current_top_k):
                                if layer in prev_top_k:
                                    prev_rank = prev_top_k.index(layer)
                                    rank_change = abs(i - prev_rank)
                                    max_rank_change = max(max_rank_change, rank_change)
                                    if rank_change > rank_tolerance:
                                        ranking_stable = False
                                else:
                                    ranking_stable = False
                                    max_rank_change = top_k

                        both_stable = energy_stable and ranking_stable

                        if both_stable:
                            stable_count += 1
                        else:
                            stable_count = 0

                        if stable_count >= patience:
                            converged = True
                            convergence_step = step_count
                            print(f"[CSPLoRA] Converged at step {step_count}")
                            break

        if cfg.probe_checkpoint and not prev_gc:
            self._toggle_gradient_checkpointing(probe_model, False)

        for p in probe_model.parameters():
            p.requires_grad = False

        torch.cuda.empty_cache()

        if step_count == 0:
            return {}, {}

        raw_scores = {k: v / step_count for k, v in running_g.items()}
        for k in self.layer_capacity:
            self.layer_capacity[k] = self.layer_capacity[k] // max(1, step_count)

        scores = self._normalize_scores(raw_scores)

        stats = {
            'actual_probe_steps': step_count,
            'converged': converged,
            'convergence_step': convergence_step,
        }

        return scores, stats

    def _compute_adaptive_rho(self, scores: Dict[str, float]) -> float:
        cfg = self.cfg
        method = cfg.rho_method.lower()

        if method == "coverage":
            return self._compute_rho_coverage(scores)
        elif method == "entropy":
            return self._compute_rho_entropy(scores)
        elif method == "gini":
            return self._compute_rho_gini(scores)
        else:
            return self._compute_rho_coverage(scores)

    def _compute_rho_coverage(self, scores: Dict[str, float]) -> float:
        cfg = self.cfg
        values = [v for v in scores.values() if v > 0]
        n = len(values)
        if n == 0:
            return cfg.rho_max

        total_energy = sum(values)
        if total_energy <= 0:
            return cfg.rho_max

        probs = [v / total_energy for v in values]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        effective_layers = math.exp(entropy)
        effective_ratio = effective_layers / n
        smoothed_ratio = math.sqrt(effective_ratio)
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * smoothed_ratio
        rho = max(cfg.rho_min, min(cfg.rho_max, rho))

        self.adaptive_stats['entropy'] = entropy
        self.adaptive_stats['effective_layers'] = effective_layers
        self.adaptive_stats['effective_ratio'] = effective_ratio
        self.adaptive_stats['rho_method'] = 'coverage'

        return rho

    def _compute_rho_entropy(self, scores: Dict[str, float]) -> float:
        cfg = self.cfg
        values = [v for v in scores.values() if v > 0]
        if not values:
            return cfg.rho_max

        total = sum(values)
        probs = [v / total for v in values]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(probs))

        if max_entropy <= 0:
            return cfg.rho_max

        norm_entropy = entropy / max_entropy
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * norm_entropy

        self.adaptive_stats['entropy'] = entropy
        self.adaptive_stats['norm_entropy'] = norm_entropy
        self.adaptive_stats['rho_method'] = 'entropy'

        return rho

    def _compute_rho_gini(self, scores: Dict[str, float]) -> float:
        cfg = self.cfg
        values = sorted([v for v in scores.values() if v > 0])
        n = len(values)
        if n == 0:
            return cfg.rho_max

        total = sum(values)
        if total <= 0:
            return cfg.rho_max

        cumsum = sum((i + 1) * v for i, v in enumerate(values))
        gini = (2 * cumsum) / (n * total) - (n + 1) / n
        gini = max(0.0, min(1.0, gini))
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * (1 - gini)

        self.adaptive_stats['gini'] = gini
        self.adaptive_stats['rho_method'] = 'gini'

        return rho

    def _allocate_ranks(self, scores: Dict[str, float], rho: float) -> Dict[str, int]:
        if not scores:
            return {}

        all_valid_items = [(name, s) for name, s in scores.items() if s > 0 and self._is_target_module(name)]
        if not all_valid_items:
            return {}

        if rho < 1.0:
            selected_items = self._select_layers_by_rho(all_valid_items, rho)
            if not selected_items:
                selected_items = all_valid_items
        else:
            selected_items = all_valid_items

        layers_data = [{"name": name, "score": s} for name, s in selected_items]

        M = len(layers_data)
        R_tot = self.cfg.R_tot

        r_min = self.cfg.r_min
        current_used = M * r_min
        remaining = R_tot - current_used

        if remaining <= 0:
            return {l['name']: r_min for l in layers_data}

        total_density = sum(l['score'] for l in layers_data)
        if total_density <= 0:
            total_density = float(M)
            for l in layers_data:
                l['score'] = 1.0

        r_max = int(self.cfg.r_max_factor * self.cfg.r_base)

        boosts = {}
        for l in layers_data:
            ratio = l['score'] / total_density
            boost = remaining * ratio
            boosts[l['name']] = boost

        final_ranks = {}
        used_boost = 0
        for l in layers_data:
            name = l['name']
            boost_int = int(boosts[name])
            final_r = min(r_min + boost_int, r_max)
            final_ranks[name] = final_r
            used_boost += (final_r - r_min)

        layers_data.sort(key=lambda x: x['score'], reverse=True)
        diff = remaining - used_boost

        idx = 0
        while diff > 0 and idx < len(layers_data):
            name = layers_data[idx]['name']
            if final_ranks[name] < r_max:
                final_ranks[name] += 1
                diff -= 1
            idx = (idx + 1) % len(layers_data)
            if idx == 0:
                all_full = all(final_ranks[n] >= r_max for n in final_ranks)
                if all_full:
                    break

        return final_ranks

    def _select_layers_by_rho(self, valid_items: List[Tuple[str, float]], rho: float) -> List[Tuple[str, float]]:
        if rho >= 1.0:
            return valid_items
        if rho <= 0:
            return []

        sorted_items = sorted(valid_items, key=lambda x: x[1], reverse=True)
        total_energy = sum(s for _, s in sorted_items)
        if total_energy <= 0:
            return valid_items

        target_energy = rho * total_energy
        cumsum = 0.0
        selected = []

        for name, s in sorted_items:
            selected.append((name, s))
            cumsum += s
            if cumsum >= target_energy:
                break

        return selected

    def _clean_layer_name(self, param_name: str) -> str:
        base = re.split(r'\.lora_[AB]', param_name)[0]
        prefixes = ["base_model.model.", "base_model.", "model."]
        for p in prefixes:
            if base.startswith(p):
                base = base[len(p):]
        if base.endswith(".weight") or base.endswith(".bias"):
            base = base.rsplit(".", 1)[0]
        return base

    def _is_target_module(self, layer_name: str) -> bool:
        return any(layer_name.endswith(tm) for tm in self.cfg.target_modules)

    def _get_module_type(self, layer_name: str) -> str:
        for tm in self.cfg.target_modules:
            if layer_name.endswith(tm):
                return tm
        return "other"

    def _smooth_density(self, density: float) -> float:
        density = max(density, 1e-12)
        smoothed = math.log1p(density)
        tau = max(self.cfg.tau_scale, 1e-6)
        smoothed = smoothed ** (1.0 / tau)
        if math.isnan(smoothed) or math.isinf(smoothed):
            return 0.0
        return smoothed

    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        if not raw_scores:
            return {}

        eps = 1e-12
        valid_items = [(n, s) for n, s in raw_scores.items() if s > 0 and self._is_target_module(n)]
        if not valid_items:
            return raw_scores

        skip_layer_norm = self.cfg.skip_layer_normalize

        type_values = defaultdict(list)
        for name, s in valid_items:
            cap = max(1, self.layer_capacity.get(name, 1))
            if skip_layer_norm:
                raw_density = max(float(s), eps)
            else:
                raw_density = max(float(s) / float(cap), eps)
            mtype = self._get_module_type(name)
            type_values[mtype].append(raw_density)

        type_means = {k: (sum(v) / len(v)) for k, v in type_values.items() if v}

        normalized = {}
        for name, s in valid_items:
            cap = max(1, self.layer_capacity.get(name, 1))
            if skip_layer_norm:
                raw_density = max(float(s), eps)
            else:
                raw_density = max(float(s) / float(cap), eps)
            mtype = self._get_module_type(name)

            if self.cfg.skip_normalize:
                final_density = raw_density
            else:
                final_density = raw_density / max(type_means.get(mtype, 1.0), eps)

            if self.cfg.skip_smooth:
                normalized[name] = final_density
            else:
                normalized[name] = self._smooth_density(final_density)

        for name, s in raw_scores.items():
            if name not in normalized:
                normalized[name] = s

        return normalized

    def _print_top_ranks(self, layer_ranks: Dict[str, int]):
        if not layer_ranks:
            return

        sorted_layers = sorted(layer_ranks.items(), key=lambda x: x[1], reverse=True)
        total_used = sum(layer_ranks.values())

        print(f"\n[CSPLoRA] Total Layers: {len(layer_ranks)}, Total Ranks: {total_used}")
        print(f"Top 10 layers:")
        for name, r in sorted_layers[:10]:
            print(f"  {name}: {r}")

    def _print_adaptive_summary(self):
        stats = self.adaptive_stats
        if not stats:
            return

        if 'actual_probe_steps' in stats:
            print(f"[CSPLoRA] Probe steps: {stats['actual_probe_steps']}, converged: {stats.get('converged', False)}")

        if 'final_rho' in stats:
            print(f"[CSPLoRA] Final rho: {stats['final_rho']:.4f}")

    def _try_load_scores(self) -> Dict[str, float] | None:
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "layer_capacity" in data:
                    self.layer_capacity = data["layer_capacity"]
                if "adaptive_stats" in data:
                    self.adaptive_stats = data["adaptive_stats"]
                return data.get("scores")
        except:
            return None

    def _save_scores(self, scores: Dict[str, float]) -> None:
        data = {
            "scores": scores,
            "layer_capacity": self.layer_capacity,
            "adaptive_stats": self.adaptive_stats,
        }
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _make_groups(self, params: List[Tuple[str, Any]], group_size: int | None) -> List[List[Tuple[str, Any]]]:
        if not params:
            return []
        if group_size is None or group_size <= 0:
            return [params]
        groups = []
        for i in range(0, len(params), group_size):
            groups.append(params[i:i + group_size])
        return groups

    def _toggle_gradient_checkpointing(self, model: nn.Module, enable: bool) -> None:
        fn = getattr(model, "gradient_checkpointing_enable", None)
        fn_disable = getattr(model, "gradient_checkpointing_disable", None)
        if enable and callable(fn):
            try:
                fn()
            except:
                pass
        if not enable and callable(fn_disable):
            try:
                fn_disable()
            except:
                pass
