# Transformer从零实现 - 英德机器翻译

本项目从零手工实现完整的Transformer模型（Encoder-Decoder架构），用于英德机器翻译任务。所有核心组件均基于PyTorch基础模块实现，不使用任何预训练的Transformer库。

## 项目特点

- 完全从零实现，不使用`torch.nn.TransformerEncoderLayer`等模块
- 实现完整的Encoder-Decoder架构
- 包含系统的消融实验，分析各组件对性能的影响
- 详细的数学推导和代码注释
- 实验完全可复现（固定随机种子seed=42）

## 项目结构

```
期中作业/
├── src/                      # 源代码
│   ├── models/              # Transformer核心模块
│   │   ├── attention.py     # Multi-Head Attention
│   │   ├── encoder.py       # Encoder
│   │   ├── decoder.py       # Decoder
│   │   ├── transformer.py   # 完整模型
│   │   ├── ffn.py          # Feed-Forward Network
│   │   ├── positional_encoding.py
│   │   └── layers.py        # LayerNorm等
│   ├── data/
│   │   └── dataset.py       # 数据处理
│   ├── utils/
│   │   ├── training.py      # 训练工具
│   │   └── visualization.py # 可视化
│   └── config.py            # 配置文件
├── scripts/
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   └── ablation_study.py   # 消融实验
├── results/
│   ├── figures/            # 实验图表
│   ├── checkpoints/        # 模型权重
│   └── logs/               # 训练日志
├── docs/
│   └── report_final.tex    # LaTeX
├── data/datasets/          # Multi30k数据集
├── requirements.txt
└── README.md
```

## 环境配置

### 激活环境

```bash
conda activate transform  
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包

```
torch==2.0.1
numpy==1.24.3
matplotlib==3.7.1
pyyaml==6.0
sentencepiece==0.1.99
tqdm==4.65.0
scikit-learn==1.3.0
```

### 硬件要求

- 推荐：NVIDIA GPU (CUDA 11.7+)，内存 8GB+
- 最低：CPU，内存 8GB+
- 训练时间：30min

## 数据集

使用Multi30k英德翻译数据集：

- 训练集：29,000 句对
- 验证集：1,014 句对
- 测试集：1,000 句对
- 数据来源：Flickr图像描述
- 平均句长：源语言约13 tokens，目标语言约12 tokens

数据集已包含在`data/datasets/`目录中。

## 快速开始


### 运行消融实验（包含baseline）

```bash
export PYTHONHASHSEED=42
python scripts/ablation_study.py  --epochs 10 --run-all
```

消融实验包括：
- Baseline（8头，4层）
- 4个注意力头
- 2个注意力头
- 2层Encoder/Decoder
- FFN维度512
- 无Dropout
- 无位置编码

### 模型评估

```bash
# 在测试集上评估
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py     --checkpoint results/checkpoints/ablation_Baseline.pth     --dataset multi30k     --use-test-set        --verbose
```

## 模型架构

完整的Transformer Encoder-Decoder架构：

### 核心组件

- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-wise Feed-Forward Network
- Positional Encoding（正弦余弦编码）
- Residual Connections + Layer Normalization (Pre-Norm)

### Encoder

- 4层Encoder Block
- 每层包含：Multi-Head Self-Attention + FFN
- 每个子层后接：Residual + LayerNorm

### Decoder

- 4层Decoder Block
- 每层包含：Masked Self-Attention + Cross-Attention + FFN
- 每个子层后接：Residual + LayerNorm

## 超参数配置

| 参数 | 值 |
|------|-----|
| Embedding维度 (d_model) | 384 |
| 注意力头数 (n_heads) | 8 |
| FFN隐藏层维度 (d_ff) | 1536 |
| Encoder层数 | 4 |
| Decoder层数 | 4 |
| Dropout | 0.25 |
| 最大序列长度 | 128 |
| Batch Size | 40 |
| 初始学习率 | 5e-3 |
| Warmup Steps | 1500 |
| 优化器 | Adam |
| Label Smoothing | 0.1 |
| 模型参数总量 | 26.9M |

## 实验结果

### 基线模型

- **训练配置**：Baseline（8头，4层，训练 10 epochs）
- **模型权重**：`results/checkpoints/ablation_Baseline.pth`
- **验证集 Loss**：3.6948（epoch 10）
- **测试集评估结果**：
  - BLEU Score：5.99
  - BLEU-1：40.15
  - BLEU-2：14.17
  - BLEU-3：5.13
  - BLEU-4：0.47

### 消融实验结果

| 配置 | 参数量 | 验证Loss | 相对基线 |
|------|--------|----------|----------|
| Baseline (8 heads, 4 layers) | 26.90M | 3.6948 | -- |
| 2 Heads | 26.90M | 3.8496 | +4.2% |
| 4 Heads | 26.90M | 3.8387 | +3.9% |
| 2 Layers | 18.62M | 3.6422 | -1.4% |
| FFN 512 | 20.60M | 3.6957 | +0.0% |
| No Dropout | 26.90M | 3.3976 | -8.0% |
| No Positional Encoding | 26.90M | 3.8911 | +5.3% |

主要发现：
- 位置编码对性能影响最大（+5.3%）
- Multi-Head机制很重要（2头比8头差4.2%）
- 在小数据集上，浅层模型（2层）略优于深层（4层）
- FFN维度存在显著冗余（从1536降到512无性能损失）
- 短训练周期下（10 epochs），无Dropout性能最好（-8.0%）

## 翻译样例

```
[1] EN: A man in an orange hat starring at something.
    DE: ein mann mit einem orangefarbenen hut starrt auf etwas.

[2] EN: People are fixing the roof of a house.
    DE: leute reparieren den dach eines hauses.

[3] EN: A guy works on a building.
    DE: ein mann arbeitet an einem gebäude.
```

## 可复现性

### 随机种子设置

所有实验使用固定随机种子42：

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

### 复现命令

```bash
# 设置环境变量
export PYTHONHASHSEED=42


#消融实验
 CUDA_VISIBLE_DEVICES=1 python scripts/ablation_study.py --run-all --epochs 10

# 评估基线模型
 CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py     --checkpoint results/checkpoints/ablation_Baseline.pth     --dataset multi30k     --use-test-set        --verbose
```

## 学术报告

完整的LaTeX位于`docs/report_final.tex`，包含：

1. 引言：Transformer背景与动机
2. 相关工作：序列模型演进
3. 模型架构与数学推导：详细公式
4. 实现细节：框架、关键代码
5. 实验设置：数据集、超参数
6. 结果与分析：训练曲线、消融实验
7. 可复现性说明：环境配置、运行命令
8. 结论与未来工作



