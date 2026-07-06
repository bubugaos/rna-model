# RNA-model 项目代码分析文档

本文档旨在为 AI 代码助手及开发者提供 `src`、`tasks` 和 `utils` 三个核心目录中所有代码文件的主要功能、工作流及其作用边界的详细说明。通过查阅此文档，可以快速了解系统架构并精准定位需要修改的代码文件，从而大幅减少阅读完整代码库带来的 token 消耗与上下文开销。

---

## 1. 项目架构概述

本项目实现了一个基于 **动态分块 (Dynamic Chunking)** 的 RNA 语言模型 (RNA LM)。模型的工作流主要如下：
1. **局部编码 (Local Encoder)**: 对输入的 RNA 序列 (A/U/C/G) 进行 Token 级别的特征提取。
2. **结构特征注入**: 结合 BPPM (碱基配对概率矩阵) 提取结构特征并注入。
3. **动态路由 (Dynamic Router)**: 利用相似度 (Attention Cosine) 和 BPPM 边界概率预测分块边界，将 Token 序列划分为可变长度的 Segment (区块)。
4. **下采样 (Downsampler)**: 提取每个分块的第一个 Token 作为该区块的全局代表特征。
5. **潜在表示 (Latent Transformer)**: 在分块级别 (Chunk-level) 运行 Transformer，学习长距离的全局依赖关系。
6. **平滑上采样 (Dechunker)**: 基于边界转移概率进行 EMA (指数滑动平均) 平滑，并将分块特征还原为 Token 级别特征。
7. **解码与训练 (Decoder & Losses)**: 结合重构头 (ReconDecoder)、任务头 (TaskDecoder) 进行掩码语言模型 (MLM) 重构或下游任务分类，计算总损失 (重构 + MLM + MDL边界损失)。

---

## 2. `src/` 核心源码目录

该目录包含模型的主干代码、数据处理流水线和训练逻辑。

### 2.1 配置文件
- **`src/config.py`**
  - **功能**: 作为项目单一的配置事实来源 (Single Source of Truth)，通过 `dataclass` 定义了 `MaskingConfig`、`ModelConfig`、`LossConfig` 和 `TrainingConfig`。包含序列化 (`save`) 和反序列化 (`from_dict`) 逻辑。
  - **修改定位**: 当你需要**添加新的超参数**（如调整默认维度、增加新的掩码概率、修改文件路径默认值等）时，修改此文件。

### 2.2 数据处理 (`src/data/`)
- **`src/data/data_loader.py`**
  - **功能**: 定义了 `RealRNADataset` 和 `collate_fn`。支持从 FASTA 读取序列，支持从 PKL 加载预计算的 BPPM，也支持通过 ViennaRNA 在线计算 (`on_the_fly`) BPPM 并缓存。处理 MLM 掩码逻辑并将其封装入 DataLoader 中。
  - **修改定位**: 当你需要**修改数据加载逻辑**、**增加新的数据特征**（如加入额外的二级结构标签）、或**更改数据对齐与 Padding (Collate) 行为**时，修改此文件。
- **`src/data/masking_utils.py`**
  - **功能**: 实现了 RNA 序列的掩码策略。包含 `SimpleMasking` (标准 BERT 随机掩码) 和 `PairingAwareMasking` (利用 BPPM 分数，倾向于同时掩码发生配对的两个碱基)。
  - **修改定位**: 当你需要**设计新的掩码任务**或**修改配对掩码逻辑**时，修改此文件。

### 2.3 模型架构 (`src/model/`)
- **`src/model/model.py`**
  - **功能**: 主模型 `RNADynamicModel` 的入口。**共享 backbone** 设计：no_chunk / fixed / dynamic 三种模式共用 `local_encoder` + BPPM 注入 + `latent_transformer` + `recon_decoder` + `classifier`，仅在中间是否插入 `router + downsampler + dechunker` 有差异。BPPM 结构注入 (`use_struct_injection`) 可独立配置，不再从属于 chunking mode，因此 ablation 可以同时控制 "是否注入结构特征" 和 "使用哪种分块策略" 两个自变量。分类池化统一为 `h_latent` 上有效位置的均值。
  - **修改定位**: 当你需要**修改整体前向传播流程 (Forward)** 或**增加全局特征融合策略**时，修改此文件。
- **`src/model/local_encoder.py`**
  - **功能**: 包含 `LocalEncoder` 类，利用 `nn.Embedding` 和 `TransformerEncoder` 提取序列的局部特征。
  - **修改定位**: 当你需要**更改底层特征提取器**（例如替换为 CNN 或 Mamba）时，修改此文件。
- **`src/model/dynamic_router.py`**
  - **功能**: 包含核心的路由逻辑 (`AttentionCosineRouter` 和 `DynamicRouter`)。融合特征相似度 (Cosine) 和结构断点 (BPPM) 得分，计算每个 Token 作为边界的概率 `boundary_probs`，并生成 `segment_ids`。`beta=0` 时通过 `use_bppm=False` 完全禁用结构项，不再有 softplus(0)≈0.693 的残留；`beta>0` 时使用 inverse softplus 初始化以保证初始值一致。**`boundary_mask` 现在通过 Vanilla STE 计算**（`hard + (probs - probs.detach())`），前向值不变、反向可将下游 loss 的梯度送回 `alpha_raw / beta_raw / logit_scale / similarity_router` 等路由器参数。注意 `segment_ids = cumsum(boundary_mask).long()` 和 Downsampler 的 `gather` 仍是非可导操作，因此在当前实现里 router 的主导学习信号走 (1) `MDL_loss → boundary_probs` 和 (2) `recon_loss → EMA → transition_probs ← boundary_probs` 两条可导路径，MDL 损失通过比较相邻 smoothed_chunks 的余弦距离自动评估边界信息增益，无需人为指定目标分块数。`compression_loss` 和 `target_chunk_ratio` 已移除。
  - **修改定位**: 当你需要**优化分块决策逻辑**、**调整边界概率计算公式**或**引入新的边界特征**时，修改此文件。
- **`src/model/chunking.py`**
  - **功能**: 包含 `Downsampler` (提取每个 Segment 第一个有效 Token 的特征) 和 `Dechunker` (基于边界预测置信度，在区块之间应用 EMA 平滑，随后通过 `segment_ids` 上采样还原为 Token 长度并残差连接)。
  - **修改定位**: 当你需要**改变特征降采样方式**（如改为 Mean Pooling）或**修改 EMA 平滑策略和特征解块上采样逻辑**时，修改此文件。
- **`src/model/latent_transformer.py`**
  - **功能**: 包含 `LatentTransformer`，接收下采样后的短序列 (Chunk-level) 并通过标准 Transformer Encoder 处理全局信息。
  - **修改定位**: 当你需要**更改全局级别的处理模型架构**时，修改此文件。
- **`src/model/decoder.py`**
  - **功能**: 包含 `ReconDecoder` (用于 MLM 重构，输出维度为词表大小) 和 `TaskDecoder` (供下游分类等任务使用)。
  - **修改定位**: 当你需要**修改输出头的网络层结构**（如增加层数、改变激活函数）时，修改此文件。
- **`src/model/positional_encoding.py`**
  - **功能**: 包含 `PositionalEncoding`，标准的正弦波位置编码实现。
  - **修改定位**: 当你需要**使用相对位置编码或旋转位置编码 (RoPE)** 代替绝对位置编码时，修改此文件。

### 2.4 训练逻辑 (`src/training/`)
- **`src/training/train.py`**
  - **功能**: 预训练的主控脚本。负责解析 CLI 参数、合并 `config.py`、初始化 DDP (分布式训练)、构建 Optimizer 和 Scheduler (Warmup + Cosine)，并执行 `train_one_epoch` 和 `evaluate` 循环。
  - **修改定位**: 当你需要**更改学习率调度器**、**增加训练日志输出/验证指标**或**修改分布式训练启动参数**时，修改此文件。
- **`src/training/losses.py`**
  - **功能**: 包含 `RNALMCriterion` 类，计算预训练的总损失。包括重构损失 (Recon)、掩码语言建模损失 (MLM) 和基于最小描述长度 (MDL) 的边界损失——通过比较相邻 smoothed_chunks 的余弦距离作为信息增益代理，结合 BPPM 茎区结构感知成本 (`mdl_delta`)，自动修剪冗余分块边界，无需人为指定 `target_chunk_ratio`。
  - **修改定位**: 当你需要**添加新的损失函数**（例如辅助对比损失）、或**修改 MDL 边界损失计算公式及防 NaN 机制**时，修改此文件。

---

## 3. `tasks/` 下游任务目录

该目录存放了基于预训练模型开展具体下游任务（如分类、评估）的脚本。

- **`tasks/finetune_classifier.py`**
  - **功能**: 专门用于 RNA 家族分类等下游任务的微调脚本。它可以对 Backbone (主干网络) 和 Head (分类头) 设置不同的学习率 (`backbone_lr_scale`)，支持加载预训练权重并进行交叉熵微调训练。**新增 `--length_buckets` CLI 选项**，val/test 评测时按长度分桶报告 accuracy / macro_f1；**未显式给 `--exp_name` 且开启了 `--no_chunk` / `--fixed_router` 时自动给 exp_name 加模式后缀**（如 `_noChunk` / `_fixed_cs8`），避免不同模式覆盖同一份 `model.pth`。
  - **修改定位**: 当你需要**调整下游分类任务的训练循环**、**增加分类任务的新指标**或**修改微调层冻结策略**时，修改此文件。
- **`tasks/evaluate_tasks.py`**
  - **功能**: 用于评估模型在分类任务和边界预测 (Boundary Diagnostics) 上的表现。**已扩展为 Boundary Ablation 与长度分层评测**：
    - `dotbracket_to_boundary(dotbracket, mode)` 支持三种真值定义：`all_transitions`、`stem_endpoints`、`stem_loop_junctions`；
    - `evaluate_boundary_ablation` 在一次 dataloader 遍历内产出 `(tolerance × truth_mode)` 的 P/R/F1 表；
    - `evaluate_classification` 接受 `length_buckets` 参数，输出 per-bucket accuracy / macro_f1 / 样本数；
    - 新增 CLI 选项 `--boundary_tolerances`、`--boundary_truth_modes`、`--length_buckets`，并写出机器可读的 `evaluate_tasks_result*_summary.json` 供后续聚合使用；
    - 同样支持 exp_name 模式后缀自动绑定，避免误读其它模式的 checkpoint。
  - **修改定位**: 当你需要**增加新的评估指标**、**修改边界容差 (Tolerance)** 或**进行离线批量评测**时，修改此文件。

---

## 4. `utils/` 工具脚本目录

该目录存放独立运行的数据预处理和辅助脚本。

- **`utils/generate_bppm.py`**
  - **功能**: 离线预处理脚本。读取 FASTA 文件，调用 ViennaRNA (`RNA.fold_compound().bpp()`) 计算每条序列的 BPPM 矩阵，并统一 Padding 后打包保存为 `.pkl` 格式。
  - **修改定位**: 当你需要**修改离线 BPPM 的生成参数**（例如更改 `max_len`，或修改截断/补零策略）时，修改此文件。
- **`utils/generate_family_dotbrackets.py`**
  - **功能**: 离线预处理脚本。调用 ViennaRNA (`RNA.fold()`) 计算 FASTA 序列的最小自由能 (MFE) 点括号二级结构，并保存为 `.txt` 文件，供 `evaluate_tasks.py` 作为边界预测的真实标签。
  - **修改定位**: 当你需要**批量生成其他结构表示**或**修改点括号输出格式**时，修改此文件。
- **`utils/stats_utils.py`**
  - **功能**: 统计推断小工具，纯 numpy 实现。提供 `bootstrap_ci(scores, ci, n_boot)` 计算样本均值的 bootstrap 置信区间，`paired_permutation_test(a, b, n_perm)` 在差异均值上做配对置换检验（默认双侧），`diff_ci(a, b, ci, n_boot)` 给出配对差异的均值 + CI。被 `utils/aggregate_results.py` 使用。
  - **修改定位**: 当你需要**新增统计量**（例如 Wilcoxon、效应量 Cohen's d）或**改变 CI/置换检验策略**时，修改此文件。
- **`utils/model_stats.py`**
  - **功能**: 模型容量诊断工具。`count_params(model)` 按 top-level 子模块返回 `params / trainable` 计数（含 `total`）；`estimate_flops(model, batch_size, seq_len)` 用 `torch.utils.flop_counter.FlopCounterMode` 在一次 dummy forward 中累计 matmul/conv FLOPs。CLI `python -m utils.model_stats [--smoke] [--output_json ...]` 会针对 no_chunk / fixed / dynamic 三模式打印对比表。
  - **修改定位**: 当你需要**报告参数量 / FLOPs**、或**比较不同模型变体的算力代价**时，修改此文件。
- **`utils/aggregate_results.py`**
  - **功能**: 多种子多模式实验聚合脚本。读取 `scripts/run_multiseed.py` 写出的 `manifest.json`，在每个 run 的 `exp_dir` 下定位 `evaluate_tasks_result*_summary.json`，按 chunking 模式聚合 mean ± std + bootstrap CI，并对每对 mode 做 paired permutation 检验，输出 `summary.json`（机器可读）和 `summary.md`（人读）。支持 `--smoke` 模式：合成一个临时 manifest + 假 summaries 自验证。
  - **修改定位**: 当你需要**新增汇总指标**（例如 boundary F1、长度分桶）或**改变报告格式**时，修改此文件。

---

## 5. `scripts/` 实验编排目录

- **`scripts/run_multiseed.py`**
  - **功能**: 多种子多模式实验驱动器。接收 `--modes` 和 `--seeds`，依次为每个 `(mode, seed)` 组合通过 `subprocess` 调用 `src.training.train`（预训练）、`tasks/finetune_classifier.py`（微调）、`tasks/evaluate_tasks.py`（评测）三步，每步独立 exp_name (`{base_name}_{mode}_seed{seed}`) 避免覆盖。所有子进程输出落到 `logs/`，并写出 `manifest.json` 供 `utils/aggregate_results.py` 消费。`--smoke` 将 `--smoke` 透传给所有子进程，几十秒内完成 orchestration 自检。
  - **修改定位**: 当你需要**改变流水线顺序**（例如插入 boundary 评测阶段）、**支持并行执行**或**新增/移除一个流水线阶段**时，修改此文件。

---

## 6. `tests/` 集中冒烟测试

- **`tests/test_smoke_pipeline.py`**
  - **功能**: 把 fix01 修复后的所有关键不变式集中验证：三模式 forward+backward 通过、STE 让 router 参数梯度非零、`classifier` 跨模式 shape 一致、`dotbracket_to_boundary` 三种真值模式语义正确、`evaluate_boundary_ablation` 的 `(mode × tolerance)` 字典完整、`_per_bucket_metrics` 长度分桶正确、`stats_utils` 在 toy 数据上行为合理、`count_params` 报告 3 模式参数。可直接 `python tests/test_smoke_pipeline.py` 跑，也兼容 pytest。
  - **修改定位**: 当你**新增任何上述模块的功能**时，请在此文件添加对应 smoke 断言，保持"修代码后 5 秒内能知道哪里坏了"的低成本验证能力。

---
**提示**：在开始修改代码前，请优先阅读此文档以确定职责边界。例如：
- 增加一种新的掩码策略：修改 `src/config.py` (加配置) -> `src/data/masking_utils.py` (加逻辑) -> `src/data/data_loader.py` (挂载逻辑)。
- 修改分块方式：仅关注 `src/model/dynamic_router.py` 和 `src/model/chunking.py`。
- 修改 Loss 比例：优先考虑通过 `src/config.py` 或 CLI 参数调整，仅在需要改变 Loss 核心公式时修改 `src/training/losses.py`。