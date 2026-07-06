# RNA-MODEL

动态 RNA 语言模型 - 基于 Transformer 的 RNA 序列建模与结构感知框架

## 项目简介

RNA-MODEL 是一个用于 RNA 序列建模的深度学习框架，采用动态分块（Dynamic Chunking）机制，能够自适应地学习 RNA 序列的语义 chunk 数量。该项目支持多种训练模式，包括预训练、微调和下游任务评估。

## 主要特性

- **动态路由机制**：自适应学习最优的分块数量
- **三种运行模式**：
  - 动态路由器 + 简单掩码（默认）
  - 固定分块路由器
  - 无分块模式
- **结构注入**：支持将 RNA 二级结构信息（BPPM）注入模型
- **多任务支持**：语言建模、序列重建、边界检测、分类任务
- **MDL 损失**：基于最小描述原理的自适应分块学习

## 环境配置

### 依赖要求

- Python 3.8+
- PyTorch
- ViennaRNA（用于 BPPM 矩阵计算）
- 详见 `requirements.txt`

### 安装 ViennaRNA

```bash
# Ubuntu/Debian
sudo apt-get install viennarna

# macOS
brew install vienna-rna

# 从源码编译
# 访问 https://www.tbi.univie.ac.at/RNA/#download
```

## 快速开始

### 数据准备

1. 预训练数据：准备 FASTA 格式的 RNA 序列
2. 预计算 BPPM 矩阵（可选）：
```bash
python -m utils.generate_bppm --fasta_path your_data.fa --output_dir ./bppm_cache
```

### 训练模型

```bash
# 预训练
python -m src.training.train --config config.yaml

# 单次实验（动态路由器 + 简单掩码）
python -m src.training.train --exp_name my_experiment

# 消融实验
python -m src.training.train --use_no_chunk true      # 无分块模式
python -m src.training.train --use_fixed_router true  # 固定分块
```

### 微调与评估

```bash
# 微调分类器
python -m tasks.finetune_classifier --checkpoint path/to/checkpoint

# 评估任务
python -m tasks.evaluate_tasks --checkpoint path/to/checkpoint
```

## 项目结构

```
RNA-MODEL/
├── src/
│   ├── config.py          # 配置管理
│   ├── data/
│   │   ├── data_loader.py # 数据加载与 BPPM 预处理
│   │   └── masking_utils.py # 掩码策略
│   ├── model/
│   │   ├── model.py       # 主模型 RNADynamicModel
│   │   ├── chunking.py    # 分块与解块模块
│   │   ├── decoder.py     # 解码器
│   │   ├── dynamic_router.py # 动态路由器
│   │   ├── latent_transformer.py # 潜在空间 Transformer
│   │   ├── local_encoder.py    # 局部编码器
│   │   └── positional_encoding.py # 位置编码
│   └── training/
│       ├── train.py       # 训练逻辑
│       └── losses.py      # 损失函数
├── tasks/
│   ├── evaluate_tasks.py  # 下游评估任务
│   └── finetune_classifier.py # 分类器微调
├── utils/
│   ├── aggregate_results.py   # 结果聚合分析
│   ├── generate_bppm.py      # BPPM 矩阵生成
│   ├── generate_family_dotbrackets.py # 结构生成
│   ├── model_stats.py        # 模型统计
│   └── stats_utils.py        # 统计工具
├── scripts/
│   └── run_multiseed.py      # 多种子实验脚本
├── data/                     # 数据目录
└── tests/                    # 单元测试
```

## 核心配置

通过 `src/config.py` 中的 `ModelConfig` 控制模型行为：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_no_chunk` | 启用无分块模式 | `false` |
| `use_fixed_router` | 使用固定分块路由器 | `false` |
| `chunk_size` | 固定分块大小 | `8` |
| `beta` | 路由器温度参数 | `1.0` |
| `use_struct_injection` | 注入结构特征 | `true` |

## 损失函数

- **MLM 损失**：掩码语言建模
- **重构损失**：序列重建
- **MDL 损失**：基于最小描述原理的自适应分块

## 工具脚本

### 多种子实验

```bash
python scripts/run_multiseed.py --mode dynamic_router --seeds 42 43 44
```

### 模型统计

```bash
python -m utils.model_stats --mode dynamic_router --embed-dim 256
```

### 结果聚合

```bash
python -m utils.aggregate_results --manifest manifest.yaml --metrics loss f1
```

## 测试

```bash
# 运行所有测试
pytest tests/

# 快速冒烟测试
pytest tests/ -v -k smoke
```

## 许可证

本项目仅供学术研究使用。