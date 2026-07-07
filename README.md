# ConProDiff: Conditional Promoter Diffusion for Controllable DNA Sequence Generation

ConProDiff 是一个基于扩散模型的条件启动子序列生成框架，能够根据指定的表达强度（低/中/高）和 GC 含量条件，生成具有生物学合理性的 DNA 启动子序列。

## 核心特性

- **双条件控制生成**：同时支持表达强度条件（low/mid/high）和 GC 含量条件的引导生成
- **Classifier-Free Guidance (CFG)**：对强度条件和 GC 条件同时执行 CFG，提升条件遵从度
- **1D UNet 扩散架构**：针对 DNA 一维序列特性设计的 UNet 去噪网络
- **多物种支持**：大肠杆菌（*E. coli*，165bp）和酿酒酵母（*S. cerevisiae*，80bp 核心序列）
- **内置 Predictor 评估**：训练过程中自动使用预训练的强度预测器进行定量评估
- **全面的分析工具**：k-mer 分布、GC 偏差分析、条件命中率、ISM 重要性分析、FIMO motif 富集等

## 项目结构

```
ConProDiff/
├── dnadiffusion/                # 核心扩散模型代码
│   ├── data/
│   │   └── dataloader.py        # 数据加载与预处理（one-hot编码、分箱、GC bin计算）
│   ├── models/
│   │   ├── diffusion.py         # 扩散过程（前向加噪、反向去噪、CFG采样）
│   │   ├── unet.py              # 1D UNet 去噪网络（双条件embedding注入）
│   │   └── layers.py            # 基础网络层（ResNetBlock、LinearAttention等）
│   └── utils/
│       ├── evaluator_diffusion.py  # 评估器（E.coli LSTM / Yeast Oracle）
│       ├── sample_util.py          # 采样工具（GC条件采样）
│       ├── train_util.py           # 训练工具（train/val step、分布式设置）
│       └── utils.py                # 通用工具函数
├── Predictor/                   # 强度预测器
│   ├── predictor_models.py      # LSTM预测模型（CNN+NonLocalBlock+LSTM）
│   └── predictor_train.py       # 预测器训练脚本
├── oracle/                      # 酵母 Oracle 预测器（DeepSTARR-style TF模型）
│   ├── predict_generated_strength.py
│   └── aux.py
├── Data/                        # 训练数据
│   ├── ecoli_exp_all.csv        # 大肠杆菌启动子数据
│   └── SC_exp_short.csv         # 酵母启动子数据
├── configs/                     # Hydra 配置文件
│   ├── train.yaml               # 训练主配置
│   ├── sample.yaml              # 采样主配置
│   ├── data/                    # 数据配置（ecoli.yaml / yeast.yaml）
│   ├── model/                   # 模型配置（unet.yaml）
│   ├── diffusion/               # 扩散配置（default.yaml）
│   ├── training/                # 训练超参（ecoli.yaml / yeast.yaml）
│   ├── sample/                  # 采样超参（ecoli.yaml / yeast.yaml）
│   └── optimizer/               # 优化器配置（adam.yaml）
├── analysis/                    # 分析脚本
├── lunwen/                      # 论文图表生成脚本
├── train.py                     # 训练入口
└── sample.py                    # 采样入口
```

## 模型架构

### 扩散模型

- **去噪网络**：1D UNet，包含 ResNet Block、Linear Attention、下采样/上采样模块
- **条件注入**：时间步 embedding + 强度 label embedding + GC bin embedding，三者相加后注入 ResNet Block
- **前向过程**：线性 beta schedule（200 步，β_start=0.0001, β_end=0.02）
- **CFG 训练**：以概率 p=0.1 同时 drop 强度条件和 GC 条件，训练无条件生成能力
- **CFG 采样**：`ε = ε_uncond + w × (ε_cond - ε_uncond)`，w 为引导强度

### 强度预测器

| 物种 | 模型 | 输入 | 说明 |
|------|------|------|------|
| E. coli | CNN + NonLocalBlock + LSTM | [B, L, 4] one-hot | PyTorch 原生推理 |
| S. cerevisiae | DeepSTARR-style TF1 | 110bp（含flank） | 通过 subprocess 调用 TF1 模型 |

### GC 条件机制

- 将每条序列的 GC fraction 按固定宽度（默认 0.02）离散化为 GC bin id
- GC bin 0 保留给 unconditional，真实 GC bin 从 1 开始
- 采样时从训练集中对应强度条件的真实 GC bin 分布中抽样，减少 GC 偏差

## 快速开始

### 环境依赖

- Python >= 3.10
- PyTorch >= 2.0
- Hydra-core >= 1.3
- NumPy, Pandas, scikit-learn
- editdistance
- TensorFlow 1.14（仅酵母 Oracle 需要，建议独立 conda 环境）

### 训练

```bash
# 训练酵母模型
python train.py

# 训练大肠杆菌模型（修改 configs/train.yaml 中的 defaults）
# 将 data: yeast 改为 data: ecoli，training: yeast 改为 training: ecoli
```

关键训练参数（在 `configs/training/` 中配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 训练 batch size |
| `num_epochs` | 300 | 最大训练轮数 |
| `cond_weight` | 2.0 | CFG 引导强度 |
| `eval_every_n_epochs` | 10 | 评估间隔 |
| `patience` | 20 | 早停耐心值 |
| `use_gc_condition` | true | 是否启用 GC 条件 |

### 采样

```bash
# 采样酵母序列
python sample.py

# 采样大肠杆菌序列（修改 configs/sample.yaml 中的 defaults）
```

采样参数（在 `configs/sample/` 中配置）：

| 参数 | 说明 |
|------|------|
| `checkpoint_path` | 模型 checkpoint 路径 |
| `number_of_samples` | 每个条件生成的序列数 |
| `guidance_scale` | CFG 引导强度 |
| `output_dir` | 输出目录 |

### 切换物种

通过修改配置文件中的 `defaults` 即可切换物种：

```yaml
# configs/train.yaml — 训练酵母
defaults:
  - data: yeast
  - training: yeast

# configs/train.yaml — 训练大肠杆菌
defaults:
  - data: ecoli
  - training: ecoli
```

## 评估指标

训练过程中自动计算的评估指标：

| 指标 | 说明 |
|------|------|
| `condition_acc` | 条件命中率（生成序列的预测强度是否落在目标区间） |
| `condition_mae` | 条件平均绝对误差 |
| `avg_corr` | k-mer 分布相关性（3/4/5-mer 的 Pearson r 均值） |
| `gc_real / gc_fake` | 真实/生成序列的 GC 含量 |
| `edit_dist` | 编辑距离 |
| `low/mid/high_hit_rate` | 各强度条件的命中率 |
| `composite_score` | 综合评估分数 |

## 分析工具

项目提供丰富的分析脚本（`analysis/` 和 `lunwen/` 目录）：

- **k-mer 分布分析**：比较生成序列与真实序列的 k-mer 频率分布
- **GC 偏差分析**：评估生成序列的 GC 含量偏差
- **ISM 重要性分析**：In Silico Mutagenesis 计算位点重要性并生成 logo 图
- **FIMO motif 富集**：使用 MEME-FIMO 扫描生成序列中的转录因子结合位点
- **t-SNE 新颖性分析**：可视化生成序列与真实序列的分布
- **消融实验**：CFG 引导强度消融、GC 条件消融

## 引用

如果您使用了本项目，请引用：

```bibtex
@article{conprodiff2026,
  title={ConProDiff: Conditional Diffusion for Controllable Promoter Design},
  author={ConProDiff Team},
  year={2026}
}
```

## 许可证

MIT License