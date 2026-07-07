import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/defined_SC_Ura_core80_812.csv"

# 新的 ResCNN + Attention 模型（our，放最后）
MY_MODEL_PATH = "/home/yt/Code/DATA/predictor_conv_transformer_attn_core80_results/best_model_by_pearson.pt"

# 旧 LSTM predictor（PromoDGDE）
LSTM_MODEL_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_SC/LSTMModel_SC_best.pth"
LSTM_SC_DATA_PATH = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/analysis_trainset_oracles_812"
os.makedirs(OUT_DIR, exist_ok=True)

# 两个 oracle 结果文件
DEFINED_ORACLE_PRED_CSV = "/home/yt/Code/DNA-Diffusion/analysis_trainset_oracles/real_with_defined_oracle_pred_strength_.csv"
COMPLEX_ORACLE_PRED_CSV = "/home/yt/Code/DNA-Diffusion/analysis_trainset_oracles/real_with_complex_oracle_pred_strength_.csv"

OUT_REAL_MYMODEL_CSV = os.path.join(OUT_DIR, "real_with_my_model_pred_strength.csv")
OUT_REAL_LSTM_CSV = os.path.join(OUT_DIR, "real_with_lstm_model_pred_strength.csv")
OUT_COMPARE_CSV = os.path.join(OUT_DIR, "trainset_real_vs_five_models_strength_summary.csv")
OUT_COMPARE_PLOT = os.path.join(OUT_DIR, "trainset_real_vs_five_models_strength_box.png")
OUT_DIST_CSV = os.path.join(OUT_DIR, "trainset_five_models_distribution_distance.csv")

SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]
LABEL_ORDER = ["low", "mid", "high"]

# 按你的要求命名，并把 our 放最后
SOURCE_ORDER = ["Real_strength", "evolution_defined", "evolution_complex", "PromoDGDE", "our"]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False
})


def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ATCG"])


def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")


def get_full_limits(series, pad_ratio=0.05, min_pad=0.01):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return None
    lo, hi = np.min(vals), np.max(vals)
    pad = max((hi - lo) * pad_ratio, min_pad) if hi > lo else min_pad
    return lo - pad, hi + pad


def load_real_strength_df_from_label(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到真实数据文件: {path}")

    real_df = pd.read_csv(path)
    real_df.columns = [c.strip().lower() for c in real_df.columns]

    if "strength" not in real_df.columns or "label" not in real_df.columns:
        raise ValueError("真实数据中缺少 strength 或 label 列")

    seq_col = find_seq_col(real_df)

    real_df = real_df.copy()
    real_df["strength"] = pd.to_numeric(real_df["strength"], errors="coerce")
    real_df["label"] = real_df["label"].astype(str).str.strip().str.lower()
    real_df["sequence_std"] = real_df[seq_col].astype(str).apply(standardize_seq)

    real_df = real_df.dropna(subset=["strength", "label", "sequence_std"]).reset_index(drop=True)
    real_df = real_df[real_df["sequence_std"].str.len() > 0]
    real_df = real_df[real_df["label"].isin(LABEL_ORDER)]
    real_df["group"] = real_df["label"].map(lambda x: f"real_{x}")
    return real_df.reset_index(drop=True)


# =========================
# 新模型：ConvTransformerPredictor
# 输入: [B, 4, 80]
# =========================
def encode_sequence_onehot_rescnn(seq: str, seq_len: int = 80) -> np.ndarray:
    arr = np.zeros((4, seq_len), dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = str(seq).strip().upper()
    for i, ch in enumerate(seq[:seq_len]):
        if ch in base_to_idx:
            arr[base_to_idx[ch], i] = 1.0
    return arr


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, C]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int, hidden: int = None):
        super().__init__()
        hidden = hidden or max(channels // 2, 32)
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        logits = self.score(x)
        attn = torch.softmax(logits, dim=1)
        pooled = (x * attn).sum(dim=1)
        return pooled


class ConvTransformerPredictor(nn.Module):
    def __init__(
        self,
        base_channels: int = 128,
        kernels=None,
        num_transformer_layers: int = 1,
        num_heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.15,
        seq_len: int = 80,
    ):
        super().__init__()
        kernels = kernels or [3, 7, 11]

        branch_out = base_channels // len(kernels)
        self.stems = nn.ModuleList([
            ConvBNAct(4, branch_out, k, dropout=dropout) for k in kernels
        ])

        stem_out = branch_out * len(kernels)

        self.proj = nn.Sequential(
            nn.Conv1d(stem_out, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(base_channels, max_len=seq_len + 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_channels,
            nhead=num_heads,
            dim_feedforward=base_channels * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.attn_pool = AttentionPooling1D(base_channels)

        self.head = nn.Sequential(
            nn.Linear(base_channels * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feats = [stem(x) for stem in self.stems]
        x = torch.cat(feats, dim=1)   # [B, stem_out, L]
        x = self.proj(x)              # [B, C, L]

        x = x.transpose(1, 2)         # [B, L, C]
        x = self.pos_encoder(x)
        x = self.transformer(x)       # [B, L, C]

        attn_feat = self.attn_pool(x)
        avg_feat = x.mean(dim=1)
        max_feat, _ = x.max(dim=1)

        feat = torch.cat([attn_feat, avg_feat, max_feat], dim=1)
        out = self.head(feat).squeeze(-1)
        return out


def predict_sequences_with_my_model(sequences, model_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = ConvTransformerPredictor(
        base_channels=int(config.get("base_channels", 128)),
        kernels=config.get("stem_kernels", [3, 7, 11]),
        num_transformer_layers=int(config.get("num_transformer_layers", 1)),
        num_heads=int(config.get("num_heads", 8)),
        ff_mult=int(config.get("ff_mult", 2)),
        dropout=float(config.get("dropout", 0.15)),
        seq_len=int(config.get("seq_len", 80)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_mean = float(ckpt["target_mean"])
    target_std = float(ckpt["target_std"])

    X = [encode_sequence_onehot_rescnn(seq, seq_len=80) for seq in sequences]
    X = np.stack(X, axis=0)  # [B, 4, 80]
    X = torch.tensor(X, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            out = model(xb).reshape(-1)
            out = out * target_std + target_mean
            preds.append(out.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1)


def attach_real_predictions_my_model(real_df, model_path):
    seqs = real_df["sequence_std"].tolist()
    preds = predict_sequences_with_my_model(seqs, model_path)

    out_df = real_df.copy()
    out_df["pred_strength_my_model"] = pd.to_numeric(preds, errors="coerce")
    out_df = out_df.dropna(subset=["pred_strength_my_model"]).reset_index(drop=True)
    out_df.to_csv(OUT_REAL_MYMODEL_CSV, index=False)
    return out_df


# =========================
# 旧模型：LSTMModel_SC_best.pth
# 输入: [B, 80, 4]
# =========================
def encode_sequence_onehot_lstm(seq: str, seq_len: int = 80) -> np.ndarray:
    arr = np.zeros((seq_len, 4), dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = str(seq).strip().upper()
    for i, ch in enumerate(seq[:seq_len]):
        if ch in base_to_idx:
            arr[i, base_to_idx[ch]] = 1.0
    return arr


def load_lstm_target_stats(sc_data_path: str):
    df = pd.read_csv(sc_data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "strength" not in df.columns:
        raise ValueError("旧 SC 数据中找不到 strength 列")
    y = pd.to_numeric(df["strength"], errors="coerce").dropna().values.astype(float)
    return float(np.mean(y)), float(np.std(y))


def predict_sequences_with_lstm_model(sequences, model_path, sc_data_path, device="cuda"):
    predictor_root = "/home/yt/Code/Bio_diffusion_gan"
    if predictor_root not in os.sys.path:
        os.sys.path.insert(0, predictor_root)

    from Predictor.predictor_models import LSTMModel

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    input_size = 4
    hidden_size = 256
    output_size = 1
    dropout_rate = 0.2
    lambda_l2 = 0.001

    model = LSTMModel(input_size, hidden_size, output_size, dropout_rate, lambda_l2).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    target_mean, target_std = load_lstm_target_stats(sc_data_path)

    X = [encode_sequence_onehot_lstm(seq, seq_len=80) for seq in sequences]
    X = np.stack(X, axis=0)
    X = torch.tensor(X, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            out = model(xb).reshape(-1)
            out = out * target_std + target_mean
            preds.append(out.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1)


def attach_real_predictions_lstm_model(real_df, model_path, sc_data_path):
    seqs = real_df["sequence_std"].tolist()
    preds = predict_sequences_with_lstm_model(seqs, model_path, sc_data_path)

    out_df = real_df.copy()
    out_df["pred_strength_lstm_model"] = pd.to_numeric(preds, errors="coerce")
    out_df = out_df.dropna(subset=["pred_strength_lstm_model"]).reset_index(drop=True)
    out_df.to_csv(OUT_REAL_LSTM_CSV, index=False)
    return out_df


# =========================
# merge defined / complex oracle
# =========================
def merge_with_defined_oracle(real_df):
    if not os.path.exists(DEFINED_ORACLE_PRED_CSV):
        raise FileNotFoundError(f"未找到 defined oracle 预测结果: {DEFINED_ORACLE_PRED_CSV}")

    df = pd.read_csv(DEFINED_ORACLE_PRED_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    seq_col = "sequence" if "sequence" in df.columns else "sequence_std"
    df["sequence_std"] = df[seq_col].astype(str).apply(standardize_seq)

    if "pred_strength_oracle" in df.columns:
        df = df.rename(columns={"pred_strength_oracle": "pred_strength_defined_oracle"})
    elif "pred_strength_defined_oracle" not in df.columns:
        raise ValueError("defined oracle 结果文件中找不到 pred_strength_oracle 或 pred_strength_defined_oracle 列")

    merged = real_df.merge(
        df[["sequence_std", "pred_strength_defined_oracle"]],
        on="sequence_std",
        how="inner"
    )
    return merged


def merge_with_complex_oracle(real_df):
    if not os.path.exists(COMPLEX_ORACLE_PRED_CSV):
        raise FileNotFoundError(f"未找到 complex oracle 预测结果: {COMPLEX_ORACLE_PRED_CSV}")

    df = pd.read_csv(COMPLEX_ORACLE_PRED_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    seq_col = "sequence" if "sequence" in df.columns else "sequence_std"
    df["sequence_std"] = df[seq_col].astype(str).apply(standardize_seq)

    if "pred_strength_complex_oracle" not in df.columns:
        if "pred_strength_oracle" in df.columns:
            df = df.rename(columns={"pred_strength_oracle": "pred_strength_complex_oracle"})
        else:
            raise ValueError("complex oracle 结果文件中找不到 pred_strength_complex_oracle 或 pred_strength_oracle 列")

    merged = real_df.merge(
        df[["sequence_std", "pred_strength_complex_oracle"]],
        on="sequence_std",
        how="inner"
    )
    return merged


# =========================
# Plot / summary
# =========================
def build_trainset_strength_plot_df(real_df):
    real_plot = real_df[["group", "strength"]].copy()
    real_plot["source"] = "Real_strength"
    real_plot["bin_level"] = real_plot["group"].str.replace("real_", "", regex=False)
    real_plot = real_plot.rename(columns={"strength": "value"})

    real_defined_plot = real_df[["group", "pred_strength_defined_oracle"]].copy()
    real_defined_plot["source"] = "evolution_defined"
    real_defined_plot["bin_level"] = real_defined_plot["group"].str.replace("real_", "", regex=False)
    real_defined_plot = real_defined_plot.rename(columns={"pred_strength_defined_oracle": "value"})

    real_complex_plot = real_df[["group", "pred_strength_complex_oracle"]].copy()
    real_complex_plot["source"] = "evolution_complex"
    real_complex_plot["bin_level"] = real_complex_plot["group"].str.replace("real_", "", regex=False)
    real_complex_plot = real_complex_plot.rename(columns={"pred_strength_complex_oracle": "value"})

    real_lstm_plot = real_df[["group", "pred_strength_lstm_model"]].copy()
    real_lstm_plot["source"] = "PromoDGDE"
    real_lstm_plot["bin_level"] = real_lstm_plot["group"].str.replace("real_", "", regex=False)
    real_lstm_plot = real_lstm_plot.rename(columns={"pred_strength_lstm_model": "value"})

    real_my_plot = real_df[["group", "pred_strength_my_model"]].copy()
    real_my_plot["source"] = "our"
    real_my_plot["bin_level"] = real_my_plot["group"].str.replace("real_", "", regex=False)
    real_my_plot = real_my_plot.rename(columns={"pred_strength_my_model": "value"})

    plot_df = pd.concat(
        [real_plot, real_defined_plot, real_complex_plot, real_lstm_plot, real_my_plot],
        ignore_index=True
    )
    plot_df["bin_level"] = pd.Categorical(plot_df["bin_level"], categories=LABEL_ORDER, ordered=True)
    plot_df["source"] = pd.Categorical(plot_df["source"], categories=SOURCE_ORDER, ordered=True)
    return plot_df


def summarize(plot_df):
    rows = []
    grouped = plot_df.groupby(["bin_level", "source"], observed=True)["value"]
    for (bin_level, source), vals in grouped:
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "bin_level": str(bin_level),
            "source": str(source),
            "count": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "q25": vals.quantile(0.25),
            "median": vals.median(),
            "q75": vals.quantile(0.75),
            "max": vals.max()
        })
    return pd.DataFrame(rows)


def plot_strength_box(plot_df, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    pos_map = {
        ("low", "Real_strength"): 0.60,
        ("low", "evolution_defined"): 0.80,
        ("low", "evolution_complex"): 1.00,
        ("low", "PromoDGDE"): 1.20,
        ("low", "our"): 1.40,

        ("mid", "Real_strength"): 1.90,
        ("mid", "evolution_defined"): 2.10,
        ("mid", "evolution_complex"): 2.30,
        ("mid", "PromoDGDE"): 2.50,
        ("mid", "our"): 2.70,

        ("high", "Real_strength"): 3.20,
        ("high", "evolution_defined"): 3.40,
        ("high", "evolution_complex"): 3.60,
        ("high", "PromoDGDE"): 3.80,
        ("high", "our"): 4.00,
    }

    color_map = {
        "Real_strength": "#0072B2",
        "evolution_defined": "#009E73",
        "evolution_complex": "#56B4E9",
        "PromoDGDE": "#CC79A7",
        "our": "#D55E00"
    }

    for b in LABEL_ORDER:
        for s in SOURCE_ORDER:
            vals = plot_df[
                (plot_df["bin_level"] == b) & (plot_df["source"] == s)
            ]["value"].dropna().values

            if len(vals) == 0:
                continue

            ax.boxplot(
                vals,
                positions=[pos_map[(b, s)]],
                widths=0.12,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=color_map[s],
                    alpha=0.30,
                    edgecolor=color_map[s],
                    linewidth=1.2
                ),
                medianprops=dict(color=color_map[s], linewidth=1.5),
                whiskerprops=dict(color=color_map[s], linewidth=0.8),
                capprops=dict(color=color_map[s], linewidth=0.8)
            )

    handles = [
        Patch(facecolor=color_map["Real_strength"], edgecolor=color_map["Real_strength"], alpha=0.5, label="Real_strength"),
        Patch(facecolor=color_map["evolution_defined"], edgecolor=color_map["evolution_defined"], alpha=0.5, label="evolution_defined"),
        Patch(facecolor=color_map["evolution_complex"], edgecolor=color_map["evolution_complex"], alpha=0.5, label="evolution_complex"),
        Patch(facecolor=color_map["PromoDGDE"], edgecolor=color_map["PromoDGDE"], alpha=0.5, label="PromoDGDE"),
        Patch(facecolor=color_map["our"], edgecolor=color_map["our"], alpha=0.5, label="our"),
    ]

    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    ax.set_xlim(0.40, 4.20)
    ax.set_xticks([1.00, 2.30, 3.60])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Strength / predicted strength")
    ax.set_title("Train-set Strength Distribution", pad=6)

    lims = get_full_limits(plot_df["value"], pad_ratio=0.06, min_pad=0.2)
    if lims:
        ax.set_ylim(*lims)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def compare_distribution_distance(real_df):
    from scipy.stats import ks_2samp, wasserstein_distance

    rows = []
    for label in LABEL_ORDER:
        sub = real_df[real_df["label"] == label]

        y_true = pd.to_numeric(sub["strength"], errors="coerce").dropna().values
        y_defined = pd.to_numeric(sub["pred_strength_defined_oracle"], errors="coerce").dropna().values
        y_complex = pd.to_numeric(sub["pred_strength_complex_oracle"], errors="coerce").dropna().values
        y_lstm = pd.to_numeric(sub["pred_strength_lstm_model"], errors="coerce").dropna().values
        y_my = pd.to_numeric(sub["pred_strength_my_model"], errors="coerce").dropna().values

        if len(y_true) and len(y_defined):
            rows.append({
                "bin": label,
                "model": "evolution_defined",
                "ks_stat": ks_2samp(y_true, y_defined).statistic,
                "wasserstein": wasserstein_distance(y_true, y_defined),
                "mean_diff": abs(np.mean(y_true) - np.mean(y_defined)),
                "std_diff": abs(np.std(y_true) - np.std(y_defined)),
            })

        if len(y_true) and len(y_complex):
            rows.append({
                "bin": label,
                "model": "evolution_complex",
                "ks_stat": ks_2samp(y_true, y_complex).statistic,
                "wasserstein": wasserstein_distance(y_true, y_complex),
                "mean_diff": abs(np.mean(y_true) - np.mean(y_complex)),
                "std_diff": abs(np.std(y_true) - np.std(y_complex)),
            })

        if len(y_true) and len(y_lstm):
            rows.append({
                "bin": label,
                "model": "PromoDGDE",
                "ks_stat": ks_2samp(y_true, y_lstm).statistic,
                "wasserstein": wasserstein_distance(y_true, y_lstm),
                "mean_diff": abs(np.mean(y_true) - np.mean(y_lstm)),
                "std_diff": abs(np.std(y_true) - np.std(y_lstm)),
            })

        if len(y_true) and len(y_my):
            rows.append({
                "bin": label,
                "model": "our",
                "ks_stat": ks_2samp(y_true, y_my).statistic,
                "wasserstein": wasserstein_distance(y_true, y_my),
                "mean_diff": abs(np.mean(y_true) - np.mean(y_my)),
                "std_diff": abs(np.std(y_true) - np.std(y_my)),
            })

    return pd.DataFrame(rows)


def main():
    real_df = load_real_strength_df_from_label(REAL_CSV)

    print("1) 预测 ConvTransformerPredictor 模型 (our)...")
    real_df = attach_real_predictions_my_model(real_df, MY_MODEL_PATH)

    print("2) 预测旧 LSTM 模型 (PromoDGDE)...")
    real_df = attach_real_predictions_lstm_model(real_df, LSTM_MODEL_PATH, LSTM_SC_DATA_PATH)

    print("3) 合并 defined oracle 结果 (evolution_defined)...")
    real_df = merge_with_defined_oracle(real_df)

    print("4) 合并 complex oracle 结果 (evolution_complex)...")
    real_df = merge_with_complex_oracle(real_df)

    print("5) 绘图与统计...")
    plot_df = build_trainset_strength_plot_df(real_df)
    summarize(plot_df).to_csv(OUT_COMPARE_CSV, index=False)
    plot_strength_box(plot_df, OUT_COMPARE_PLOT)
    print(f"训练集强度分布图已保存: {OUT_COMPARE_PLOT}")

    dist_df = compare_distribution_distance(real_df)
    dist_df.to_csv(OUT_DIST_CSV, index=False)
    print(f"分布距离表已保存: {OUT_DIST_CSV}")
    print(dist_df)


if __name__ == "__main__":
    main()