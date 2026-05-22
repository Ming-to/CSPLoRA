# CSPLoRA

本仓库是 **CSPLoRA** 的官方实现。论文已被 **ICML 2026** 接收。

CSPLoRA 是一种面向参数高效微调的置信度感知结构化秩分配方法。与对所有模块使用相同 LoRA rank 的做法不同，CSPLoRA 会先通过轻量级 probe 阶段估计不同模块的重要性，再在给定参数预算下分配 LoRA rank。rank 分配确定后，正式训练阶段仍然遵循所选 LoRA 后端的训练流程。

## 简介

CSPLoRA 主要关注以下问题：

- 如何根据具体任务和模型结构，为不同模块分配更合适的 LoRA rank。
- 如何在 probe 阶段利用置信度信息，减弱噪声样本对重要性估计的影响。
- 如何保存并复用 rank 分配结果，减少重复 probe 带来的额外开销。

## 仓库结构

```text
.
├── csplora.py                  # CSPLoRA 核心实现
├── train_*.py                  # 不同任务的训练入口
├── *_args.py                   # 参数配置
├── scripts/                    # 运行脚本示例
├── instruction_tuning_eval/    # 指令微调相关评测脚本
├── utils/                      # 数据处理与通用工具
└── requirements.txt            # Python 依赖
```

## 环境安装

```bash
conda create -n csplora python=3.10
conda activate csplora
pip install -r requirements.txt
```

需要根据本地 GPU 和 CUDA 环境安装匹配版本的 PyTorch。

## 快速开始

请先在本地准备好预训练模型和数据集，并根据自己的路径修改 `scripts/` 下的脚本。

常用运行入口如下：

```bash
bash scripts/run_cr.sh          # Commonsense reasoning
bash scripts/run_arithmetic.sh  # Arithmetic reasoning
bash scripts/run_glue.sh        # GLUE
```

运行 CSPLoRA 时，在对应脚本中启用 `--csplora` 参数；运行普通 LoRA baseline 时，关闭 `--csplora` 并使用统一的 LoRA rank 配置。

## 置信度加权

默认 CSPLoRA 使用固定的置信度加权：

```bash
--csplora_gamma_strategy fixed
--csplora_gamma 2.0
```

仓库中也提供了自适应 gamma 选项：

```bash
--csplora \
  --csplora_gamma_strategy adaptive_std \
  --csplora_gamma_scale 1.0 \
  --csplora_gamma_min_std 1e-4
```

gamma 相关设置只影响 probe 和 rank 规划阶段。cache 文件名中会记录 gamma 策略，避免不同 probe 设置之间误用缓存。

## 引用

如果本仓库对你的研究有帮助，请引用我们的论文：

```bibtex
@inproceedings{
anonymous2026csplora,
title={{CSPL}o{RA}: Confidence-Guided Structure Planning for Low-Rank Adaptation},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=WkARznT46l}
}
```
