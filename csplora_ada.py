"""
C-SPLoRA with Adaptive Step and Rho (top-p) Selection

This module extends CSpLoRAPlanner with:
1. Adaptive probe steps based on convergence detection
2. Adaptive rho (top-p) selection based on energy coverage

Convergence Detection Strategy (adapted from GoRA 3.3 for C-SPLoRA):
- Primary: Total gradient energy relative change < ε
- Secondary: Top-K layer ranking stability (no major rank swaps)
- Require both conditions for `patience` consecutive checks

Key differences from pure GoRA:
- C-SPLoRA uses confidence-weighted loss (gamma), which affects gradient dynamics
- We monitor normalized per-layer scores, not just raw energy
- Rho is determined by cumulative energy coverage (what % of layers cover X% energy)
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
class AdaptiveCSpLoRAConfig:
    """Configuration for Adaptive C-SPLoRA with auto-tuned step and rho."""
    model_id: str
    task_name: str
    target_modules: List[str]
    task_type: TaskType
    cache_dir: str

    # Core hyperparameters (same as original)
    gamma: float = 1.0
    r_base: int = 8
    r_min: int = 2
    r_max_factor: float = 4.0
    tau_scale: float = 1.0
    skip_smooth: bool = False
    skip_normalize: bool = False
    R_tot: int | None = None
    device: str = "cuda"

    # Probe settings
    probe_r: int = 2
    probe_alpha: int = 8
    probe_lr: float = 5e-4
    probe_mode: str = "full_lowmem"
    probe_use_amp: bool = True
    probe_group_size: int | None = None
    probe_checkpoint: bool = True  # Enable gradient checkpointing during probe to reduce memory

    # ========== Adaptive Step Settings (C-SPLoRA specific) ==========
    adaptive_steps: bool = True          # Enable adaptive step selection
    min_probe_steps: int = 50            # Minimum probe steps
    max_probe_steps: int = 500           # Maximum probe steps
    step_convergence_threshold: float = 0.02  # Energy relative change threshold (2%)
    step_check_interval: int = 10        # Check convergence every N steps
    step_patience: int = 3               # Require N consecutive stable checks
    step_top_k: int = 20                 # Monitor top-K layers for ranking stability
    step_rank_tolerance: int = 5         # Allow rank changes up to this amount in top-K

    # ========== Adaptive Rho Settings (Energy Coverage based) ==========
    adaptive_rho: bool = True            # Enable adaptive rho selection
    rho_min: float = 0.6                 # Minimum coverage ratio
    rho_max: float = 1.0                 # Maximum coverage ratio
    rho_method: str = "coverage"         # Method: "coverage" (recommended), "entropy", "gini"
    rho_coverage_target: float = 0.95    # Target energy coverage for "coverage" method
    rho_fixed: float = 0.9               # Fixed rho when adaptive_rho=False

    # ========== Importance Metric Settings ==========
    importance_metric: str = "taylor"    # "taylor" (default), "fisher", "gora"
    # taylor: E = ||g * W||_1 = sum(|g * W|)  -- Taylor expansion (default, best empirical)
    # fisher: E = ||g||_F^2 = sum(g^2)  -- diagonal Fisher trace
    # gora:   E = mean(|g * W|)  -- GoRA-style importance (normalized taylor)

    # Layer-wise normalization (÷ numel) control
    skip_layer_normalize: bool = True    # Default: skip layer normalization (Taylor already accounts for scale)

    # Legacy compatibility
    max_probe_steps_legacy: int = 200    # For non-adaptive mode
    probe_coverage: float = 0.0
    probe_ensemble: int = 1
    probe_steps_per_group: int | None = None


class AdaptiveCSpLoRAPlanner:
    """
    C-SPLoRA Planner with adaptive step and rho selection.

    Key innovations:
    1. Adaptive probe steps: Monitor convergence of layer importance rankings
       and stop early when stable (inspired by GoRA Section 3.3)
    2. Adaptive rho: Use entropy of importance distribution to determine
       optimal layer coverage ratio
    """

    def __init__(self, cfg: AdaptiveCSpLoRAConfig):
        self.cfg = cfg
        self.layer_capacity: Dict[str, int] = {}
        self.convergence_history: List[Dict[str, float]] = []
        self.adaptive_stats: Dict[str, Any] = {}

        raw_id = os.path.basename(cfg.model_id)
        safe_model_id = raw_id.replace("/", "_").replace(":", "_")

        # Include importance_metric in cache filename to avoid mixing different metrics
        metric_suffix = f"_{cfg.importance_metric}" if cfg.importance_metric != "taylor" else ""
        fname = f"{cfg.task_name}_ada_scores{metric_suffix}.json"
        self.cache_path = os.path.join(cfg.cache_dir, safe_model_id, fname)

    def plan(self, base_model: nn.Module, probe_dataloader) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Main planning entry point."""
        scores = self._try_load_scores()

        if scores is None:
            print(f"[AdaCSpLoRA] No cached scores found at {self.cache_path}. Running adaptive probe...")
            scores, probe_stats = self._run_adaptive_probe(base_model, probe_dataloader)
            self.adaptive_stats.update(probe_stats)
            self._save_scores(scores)
        else:
            print(f"[AdaCSpLoRA] Loaded scores from {self.cache_path}")

        # Filter valid target modules
        valid_scores = {k: v for k, v in scores.items() if v > 0 and self._is_target_module(k)}
        num_potential_layers = len(valid_scores)
        if num_potential_layers == 0:
            print("[AdaCSpLoRA] No valid target modules found in probe scores.")
            return {}, scores

        # Calculate budget
        if self.cfg.R_tot is None or self.cfg.R_tot <= 0:
            self.cfg.R_tot = num_potential_layers * self.cfg.r_base
            print(f"[AdaCSpLoRA] Auto-calculated Budget: {self.cfg.R_tot} (Layers: {num_potential_layers})")
        else:
            print(f"[AdaCSpLoRA] Using User-Defined Fixed Budget: {self.cfg.R_tot}")

        # Determine rho (either adaptive or fixed)
        rho = self._compute_adaptive_rho(valid_scores) if self.cfg.adaptive_rho else self.cfg.rho_fixed
        self.adaptive_stats['final_rho'] = rho

        # Allocate ranks
        layer_ranks = self._allocate_ranks(valid_scores, rho)
        self._print_top_ranks(layer_ranks)
        self._print_adaptive_summary()

        return layer_ranks, scores

    # ========== Adaptive Probe with Hybrid Convergence Detection ==========

    def _run_adaptive_probe(self, base_model: nn.Module, probe_dataloader) -> Tuple[Dict[str, float], Dict]:
        """
        Run probe with hybrid adaptive step selection using full_lowmem strategy.

        Uses group-based gradient computation on original model weights to reduce memory.
        Only activates gradients for one group of parameters at a time.

        Convergence Strategy (adapted for C-SPLoRA's confidence-weighted loss):
        1. Primary condition: Total gradient energy relative change < ε
        2. Secondary condition: Top-K layer rankings are stable (no major rank swaps)
        3. Converge when BOTH conditions are met for `patience` consecutive checks
        """
        cfg = self.cfg
        probe_model = base_model
        probe_model.to(cfg.device)
        probe_model.eval()

        prev_gc = getattr(probe_model, "is_gradient_checkpointing", False)
        if cfg.probe_checkpoint:
            self._toggle_gradient_checkpointing(probe_model, True)

        # Freeze all parameters first
        for p in probe_model.parameters():
            p.requires_grad = False

        # Find target parameters (original weights, not LoRA)
        target_params = []
        for name, param in probe_model.named_parameters():
            if "lora_" in name or "adapter" in name:
                continue
            clean_name = self._clean_layer_name(name)
            if self._is_target_module(clean_name):
                target_params.append((name, param))

        if not target_params:
            print("[AdaCSpLoRA] Full-lowmem probe: no target params matched target_modules.")
            if cfg.probe_checkpoint and not prev_gc:
                self._toggle_gradient_checkpointing(probe_model, False)
            return {}, {}

        # Group parameters for memory-efficient probing
        groups = self._make_groups(target_params, cfg.probe_group_size)
        amp_enabled = cfg.probe_use_amp and torch.cuda.is_available() and ("cuda" in cfg.device)

        # Initialize accumulators
        running_g = defaultdict(float)
        step_count = 0
        converged = False
        convergence_step = -1

        # Convergence tracking
        energy_history: List[float] = []
        ranking_history: List[List[str]] = []  # Store top-K layer rankings at each checkpoint
        stable_count = 0  # Count consecutive stable checks

        min_steps = cfg.min_probe_steps if cfg.adaptive_steps else cfg.max_probe_steps_legacy
        max_steps = cfg.max_probe_steps if cfg.adaptive_steps else cfg.max_probe_steps_legacy
        patience = cfg.step_patience
        top_k = cfg.step_top_k
        rank_tolerance = cfg.step_rank_tolerance

        # Calculate steps per group
        steps_per_group = cfg.probe_steps_per_group or max_steps

        print(f"[AdaCSpLoRA] Full-lowmem probe: groups={len(groups)}, steps/group={steps_per_group}, "
              f"min={min_steps}, max={max_steps}, threshold={cfg.step_convergence_threshold}, "
              f"patience={patience}, top_k={top_k}, rank_tolerance={rank_tolerance}, "
              f"metric={cfg.importance_metric}, AMP={amp_enabled}")

        for gi, group in enumerate(groups):
            if converged:
                break

            active_names = {n for n, _ in group}
            # Only enable gradients for current group
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

                # Collect gradients from current group
                for name, param in group:
                    if param.grad is None:
                        continue
                    clean_name = self._clean_layer_name(name)
                    grad = param.grad.detach()

                    # Compute importance score based on selected metric
                    metric = cfg.importance_metric.lower()
                    if metric == "fisher":
                        # Fisher: E = ||g||_F^2 = sum(g^2)
                        g_score = (grad * grad).sum().item()
                    elif metric == "taylor":
                        # Taylor: E = ||g * W||_1 = sum(|g * W|)
                        weight = param.data.detach()
                        g_score = (grad.abs() * weight.abs()).sum().item()
                    elif metric == "gora":
                        # GoRA: E = mean(|g * W|)
                        weight = param.data.detach()
                        g_score = (grad.abs() * weight.abs()).mean().item()
                    else:
                        # Default to Fisher
                        g_score = (grad * grad).sum().item()

                    if math.isnan(g_score) or math.isinf(g_score):
                        continue
                    running_g[clean_name] += g_score
                    self.layer_capacity[clean_name] = self.layer_capacity.get(clean_name, 0) + param.numel()

                step_count += 1

                # Hybrid convergence check (only after processing all groups at least once)
                total_steps_per_round = len(groups) * steps_per_group
                if cfg.adaptive_steps and step_count >= min_steps and gi == len(groups) - 1:
                    if step_count % cfg.step_check_interval == 0:
                        # Condition 1: Total energy relative change
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

                        # Condition 2: Top-K layer ranking stability
                        current_raw = {k: v / step_count for k, v in running_g.items()}
                        current_capacity = {k: self.layer_capacity.get(k, 1) // max(1, step_count)
                                           for k in current_raw.keys()}

                        # Compute density = raw_score / capacity
                        current_density = {}
                        for k, raw_s in current_raw.items():
                            cap = max(1, current_capacity.get(k, 1))
                            current_density[k] = raw_s / cap

                        # Sort by density to get true importance ranking
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
                            print(f"[AdaCSpLoRA] Step {step_count}: δ={delta:.4f}, max_rank_change={max_rank_change} "
                                  f"(stable {stable_count}/{patience})")
                        else:
                            stable_count = 0
                            if step_count % 50 == 0:
                                print(f"[AdaCSpLoRA] Step {step_count}: δ={delta:.4f}, max_rank_change={max_rank_change} "
                                      f"(energy_stable={energy_stable}, ranking_stable={ranking_stable})")

                        if stable_count >= patience:
                            converged = True
                            convergence_step = step_count
                            print(f"[AdaCSpLoRA] Convergence detected at step {step_count} "
                                  f"(both energy and ranking stable for {patience} consecutive checks)")
                            break

        # Cleanup: disable gradient checkpointing if we enabled it
        if cfg.probe_checkpoint and not prev_gc:
            self._toggle_gradient_checkpointing(probe_model, False)

        # Freeze all parameters again
        for p in probe_model.parameters():
            p.requires_grad = False

        torch.cuda.empty_cache()

        if step_count == 0:
            return {}, {}

        raw_scores = {k: v / step_count for k, v in running_g.items()}
        for k in self.layer_capacity:
            self.layer_capacity[k] = self.layer_capacity[k] // max(1, step_count)

        # Normalize scores
        scores = self._normalize_scores(raw_scores)

        # Collect stats
        stats = {
            'actual_probe_steps': step_count,
            'converged': converged,
            'convergence_step': convergence_step,
            'adaptive_steps_enabled': cfg.adaptive_steps,
            'energy_history': energy_history[-10:] if energy_history else [],
            'final_stable_count': stable_count,
            'final_top_k': ranking_history[-1] if ranking_history else [],
            'num_groups': len(groups),
            'steps_per_group': steps_per_group,
        }

        print(f"[AdaCSpLoRA] Probe finished: {step_count} steps, converged={converged}")
        return scores, stats

    # ========== Adaptive Rho Selection ==========

    def _compute_adaptive_rho(self, scores: Dict[str, float]) -> float:
        """Compute adaptive rho based on importance distribution."""
        cfg = self.cfg
        method = cfg.rho_method.lower()

        if method == "coverage":
            return self._compute_rho_coverage(scores)
        elif method == "entropy":
            return self._compute_rho_entropy(scores)
        elif method == "gini":
            return self._compute_rho_gini(scores)
        elif method == "elbow":
            return self._compute_rho_elbow(scores)
        else:
            print(f"[AdaCSpLoRA] Unknown rho method '{method}', using coverage")
            return self._compute_rho_coverage(scores)

    def _compute_rho_coverage(self, scores: Dict[str, float]) -> float:
        """
        Compute rho based on effective number of layers (information-theoretic approach).

        This method uses entropy to compute the "effective number of layers" that
        contribute meaningfully to the gradient energy, then maps it to a coverage target.

        Logic:
        1. Normalize scores to probability distribution
        2. Compute entropy H = -Σ p_i * log(p_i)
        3. Effective layers = exp(H) (perplexity)
        4. Compute rho as coverage target based on effective_layers / total_layers ratio

        Key insight:
        - If energy is concentrated in few layers: low entropy → few effective layers → lower rho
        - If energy is spread uniformly: high entropy → many effective layers → higher rho

        The rho returned is an ENERGY COVERAGE TARGET (not layer ratio), which will be
        used by _select_layers_by_rho to select layers covering that % of total energy.
        """
        cfg = self.cfg
        values = [v for v in scores.values() if v > 0]
        n = len(values)
        if n == 0:
            return cfg.rho_max

        total_energy = sum(values)
        if total_energy <= 0:
            return cfg.rho_max

        # Normalize to probability distribution
        probs = [v / total_energy for v in values]

        # Compute entropy
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(n)  # Maximum entropy for uniform distribution

        # Effective number of layers = exp(entropy), i.e., perplexity
        effective_layers = math.exp(entropy)

        # Ratio of effective layers to total layers [0, 1]
        effective_ratio = effective_layers / n

        # Map effective_ratio to rho (energy coverage target)
        # effective_ratio 低 → 能量集中 → 可以用较低的覆盖目标
        # effective_ratio 高 → 能量分散 → 需要较高的覆盖目标
        #
        # 使用平滑映射：rho = rho_min + (rho_max - rho_min) * effective_ratio
        # 但我们希望即使 effective_ratio 较低，rho 也不能太低，所以用 sqrt 平滑
        smoothed_ratio = math.sqrt(effective_ratio)  # 使映射更平缓
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * smoothed_ratio

        # Clamp to allowed range
        rho = max(cfg.rho_min, min(cfg.rho_max, rho))

        # Store stats
        self.adaptive_stats['entropy'] = entropy
        self.adaptive_stats['max_entropy'] = max_entropy
        self.adaptive_stats['norm_entropy'] = entropy / max_entropy if max_entropy > 0 else 0
        self.adaptive_stats['effective_layers'] = effective_layers
        self.adaptive_stats['effective_ratio'] = effective_ratio
        self.adaptive_stats['total_layers'] = n
        self.adaptive_stats['rho_method'] = 'coverage'

        print(f"[AdaCSpLoRA] Effective-layers rho: entropy={entropy:.4f}, "
              f"effective_layers={effective_layers:.1f}/{n}, "
              f"ratio={effective_ratio:.4f}, rho={rho:.4f}")

        return rho

    def _compute_rho_entropy(self, scores: Dict[str, float]) -> float:
        """
        Compute rho based on entropy of importance distribution.

        High entropy (uniform) -> large rho (keep more layers)
        Low entropy (concentrated) -> small rho (focus on important layers)
        """
        cfg = self.cfg
        values = [v for v in scores.values() if v > 0]
        if not values:
            return cfg.rho_max

        # Normalize to probability distribution
        total = sum(values)
        probs = [v / total for v in values]

        # Compute entropy
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(probs))  # Maximum entropy for uniform distribution

        if max_entropy <= 0:
            return cfg.rho_max

        # Normalized entropy [0, 1]
        norm_entropy = entropy / max_entropy

        # Map to rho range: high entropy -> high rho
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * norm_entropy

        self.adaptive_stats['entropy'] = entropy
        self.adaptive_stats['norm_entropy'] = norm_entropy
        self.adaptive_stats['rho_method'] = 'entropy'

        print(f"[AdaCSpLoRA] Entropy-based rho: entropy={entropy:.4f}, norm={norm_entropy:.4f}, rho={rho:.4f}")
        return rho

    def _compute_rho_gini(self, scores: Dict[str, float]) -> float:
        """
        Compute rho based on Gini coefficient of importance distribution.

        High Gini (unequal) -> small rho (focus on important layers)
        Low Gini (equal) -> large rho (keep more layers)
        """
        cfg = self.cfg
        values = sorted([v for v in scores.values() if v > 0])
        n = len(values)
        if n == 0:
            return cfg.rho_max

        # Gini coefficient
        total = sum(values)
        if total <= 0:
            return cfg.rho_max

        cumsum = sum((i + 1) * v for i, v in enumerate(values))
        gini = (2 * cumsum) / (n * total) - (n + 1) / n

        # Clamp to [0, 1]
        gini = max(0.0, min(1.0, gini))

        # Map: high Gini (unequal) -> low rho
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * (1 - gini)

        self.adaptive_stats['gini'] = gini
        self.adaptive_stats['rho_method'] = 'gini'

        print(f"[AdaCSpLoRA] Gini-based rho: gini={gini:.4f}, rho={rho:.4f}")
        return rho

    def _compute_rho_elbow(self, scores: Dict[str, float]) -> float:
        """
        Compute rho by finding the elbow point in sorted importance scores.
        """
        cfg = self.cfg
        sorted_scores = sorted(scores.values(), reverse=True)
        n = len(sorted_scores)
        if n == 0:
            return cfg.rho_max

        total = sum(sorted_scores)
        if total <= 0:
            return cfg.rho_max

        # Find elbow: where the gap exceeds average gap significantly
        gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
        if not gaps:
            return cfg.rho_max

        avg_gap = sum(gaps) / len(gaps)
        threshold = avg_gap * 1.5  # Gap significantly larger than average

        elbow_idx = n  # Default: keep all
        for i, gap in enumerate(gaps):
            if gap > threshold:
                elbow_idx = i + 1
                break

        # Compute rho as cumulative coverage at elbow
        cumsum = sum(sorted_scores[:elbow_idx])
        rho = cumsum / total

        # Clamp to range
        rho = max(cfg.rho_min, min(cfg.rho_max, rho))

        self.adaptive_stats['elbow_idx'] = elbow_idx
        self.adaptive_stats['elbow_layers'] = elbow_idx
        self.adaptive_stats['rho_method'] = 'elbow'

        print(f"[AdaCSpLoRA] Elbow-based rho: elbow_idx={elbow_idx}/{n}, rho={rho:.4f}")
        return rho

    # ========== Rank Allocation (same as original, with explicit rho) ==========

    def _allocate_ranks(self, scores: Dict[str, float], rho: float) -> Dict[str, int]:
        """Rank allocation with explicit rho parameter."""
        if not scores:
            return {}

        all_valid_items = [(name, s) for name, s in scores.items() if s > 0 and self._is_target_module(name)]
        if not all_valid_items:
            return {}

        # Layer selection using rho
        if rho < 1.0:
            selected_items = self._select_layers_by_rho(all_valid_items, rho)
            if not selected_items:
                print("[AdaCSpLoRA] Warning: rho selection resulted in 0 layers. Using all layers.")
                selected_items = all_valid_items
        else:
            selected_items = all_valid_items

        layers_data = [{"name": name, "score": s} for name, s in selected_items]

        M = len(layers_data)
        R_tot = self.cfg.R_tot

        r_min = self.cfg.r_min
        current_used = M * r_min
        remaining = R_tot - current_used

        strategy_name = f"rho={rho:.3f} Selection + " if rho < 1.0 else ""
        print(f"[AdaCSpLoRA] Strategy: {strategy_name}Baseline({r_min}) + Linear Boost")
        print(f"[AdaCSpLoRA] Selected Layers: {M}, Remaining Budget for Boosting: {remaining} (Total: {R_tot})")

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

        # Rebalancing
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
        """Select layers using top-p cumulative energy coverage."""
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

        print(f"[AdaCSpLoRA] rho={rho:.3f} layer selection: {len(selected)}/{len(valid_items)} layers "
              f"(covering {cumsum/total_energy*100:.1f}% energy)")

        return selected

    # ========== Helper Methods ==========

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
        """Normalize raw gradient scores."""
        if not raw_scores:
            return {}

        eps = 1e-12
        valid_items = [(n, s) for n, s in raw_scores.items() if s > 0 and self._is_target_module(n)]
        if not valid_items:
            return raw_scores

        # Check if we should skip layer-wise normalization
        skip_layer_norm = self.cfg.skip_layer_normalize

        type_values = defaultdict(list)
        for name, s in valid_items:
            cap = max(1, self.layer_capacity.get(name, 1))
            # Layer-wise normalization (÷ cap): convert sum → mean
            if skip_layer_norm:
                raw_density = max(float(s), eps)  # Skip ÷cap for ablation
            else:
                raw_density = max(float(s) / float(cap), eps)
            mtype = self._get_module_type(name)
            type_values[mtype].append(raw_density)

        type_means = {k: (sum(v) / len(v)) for k, v in type_values.items() if v}

        normalized = {}
        for name, s in valid_items:
            cap = max(1, self.layer_capacity.get(name, 1))
            # Layer-wise normalization
            if skip_layer_norm:
                raw_density = max(float(s), eps)
            else:
                raw_density = max(float(s) / float(cap), eps)
            mtype = self._get_module_type(name)

            # Type normalization (÷ type_mean)
            if self.cfg.skip_normalize:
                final_density = raw_density
            else:
                final_density = raw_density / max(type_means.get(mtype, 1.0), eps)

            # Smoothing (log1p)
            if self.cfg.skip_smooth:
                normalized[name] = final_density
            else:
                normalized[name] = self._smooth_density(final_density)

        for name, s in raw_scores.items():
            if name not in normalized:
                normalized[name] = s

        layer_norm_status = "skip_layer_norm" if skip_layer_norm else "layer_normalized"
        type_norm_status = "skip_type_norm" if self.cfg.skip_normalize else "type_normalized"
        smooth_status = "skip_smooth" if self.cfg.skip_smooth else "smoothed"
        print(f"[AdaCSpLoRA] Processed {len(valid_items)} scores "
              f"({layer_norm_status}, {type_norm_status}, {smooth_status}).")
        return normalized

    def _print_top_ranks(self, layer_ranks: Dict[str, int]):
        if not layer_ranks:
            return

        sorted_layers = sorted(layer_ranks.items(), key=lambda x: x[1], reverse=True)
        total_used = sum(layer_ranks.values())

        print("\n" + "=" * 60)
        print(f"[AdaCSpLoRA] Rank Allocation Summary")
        print(f"Total Selected Layers: {len(layer_ranks)}")
        print(f"Total Ranks Used: {total_used} / Budget: {self.cfg.R_tot}")
        print(f"Max Rank: {max(layer_ranks.values())} | Min Rank: {min(layer_ranks.values())}")
        print("-" * 60)
        print(f"{'Layer Name':<50} | {'Rank':<5} | {'% of Budget':<10}")
        print("-" * 60)

        for name, r in sorted_layers[:15]:
            ratio = (r / total_used) * 100
            print(f"{name:<50} | {r:<5} | {ratio:.1f}%")

        if len(sorted_layers) > 15:
            print(f"... and {len(sorted_layers) - 15} more layers.")
        print("=" * 60 + "\n")

    def _print_adaptive_summary(self):
        """Print summary of adaptive decisions."""
        stats = self.adaptive_stats
        if not stats:
            return

        print("\n" + "=" * 60)
        print("[AdaCSpLoRA] Adaptive Selection Summary")
        print("-" * 60)

        if 'actual_probe_steps' in stats:
            print(f"Probe Steps: {stats['actual_probe_steps']} "
                  f"(converged={stats.get('converged', False)}, "
                  f"at step {stats.get('convergence_step', 'N/A')})")

        if 'final_rho' in stats:
            method = stats.get('rho_method', 'unknown')
            print(f"Final Rho: {stats['final_rho']:.4f} (method: {method})")

            if method == 'entropy':
                print(f"  - Entropy: {stats.get('entropy', 'N/A'):.4f}")
                print(f"  - Normalized Entropy: {stats.get('norm_entropy', 'N/A'):.4f}")
            elif method == 'gini':
                print(f"  - Gini Coefficient: {stats.get('gini', 'N/A'):.4f}")
            elif method == 'elbow':
                print(f"  - Elbow Index: {stats.get('elbow_idx', 'N/A')}")

        print("=" * 60 + "\n")

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
        meta = {
            "model": self.cfg.model_id,
            "task": self.cfg.task_name,
            "adaptive_steps": self.cfg.adaptive_steps,
            "adaptive_rho": self.cfg.adaptive_rho,
            "rho_method": self.cfg.rho_method,
        }
        data = {
            "meta": meta,
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
