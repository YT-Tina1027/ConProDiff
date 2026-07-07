import os
import sys
import gc
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from collections import Counter
from itertools import product
from matplotlib.patches import Patch
import seaborn as sns
import matplotlib as mpl

# ================== 静音设置 ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ================== 全局绘图风格（Nature 论文级） ==================
plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             10,
    "axes.labelsize":        11,
    "axes.titlesize":        11,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "legend.fontsize":       9,
    "legend.title_fontsize": 10,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "axes.unicode_minus":    False,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
})
_RC = dict(plt.rcParams)
np.random.seed(42)

# =========================================================
# 路径配置：你只需要改这里
# =========================================================

GEN_LOW_TXT = "/home/yt/Code/DNA-Diffusion/result/ecoli_gclx/low.txt"
GEN_MID_TXT = "/home/yt/Code/DNA-Diffusion/result/ecoli_gclx/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/ecoli_gclx/high.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
TEST_REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/ecoli_test_3000.csv"

# 当前训练使用的数据处理结果
PKL_PATH = "/home/yt/Code/DNA-Diffusion/processed_data/ecoli_diffusion_data_gc_lx.pkl"

PREDICTOR_MODEL_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/ecoli_gclx"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_GENERATED_PRED_CSV = os.path.join(OUT_DIR, "generated_with_pred_strength.csv")
OUT_REAL_TEST_PRED_CSV = os.path.join(OUT_DIR, "real_test_with_pred_strength.csv")
# ✅ 新增：全集 oracle 预测结果保存路径，避免覆盖测试集预测文件
OUT_REAL_ALL_PRED_CSV  = os.path.join(OUT_DIR, "real_all_with_oracle_pred_strength.csv")

OUT_COMPARE_CSV = os.path.join(OUT_DIR, "real_vs_generated_vs_realpred_strength_summary.csv")
OUT_COMPARE_PLOT = os.path.join(OUT_DIR, "real_vs_generated_vs_realpred_strength_box.png")

OUT_GC_CSV = os.path.join(OUT_DIR, "gc_content_summary.csv")
OUT_GC_PLOT = os.path.join(OUT_DIR, "gc_content_box.png")

OUT_KMER_PCC_CSV = os.path.join(OUT_DIR, "kmer_345_pcc_summary.csv")
OUT_KMER_PLOT = os.path.join(OUT_DIR, "kmer_345_pcc_bar.png")

# ✅ 彻底设置为 None，全面启用全自动自适应等频分箱逻辑
TEST_BIN_EDGES_OVERRIDE = None

# =========================================================
# 全局常量
# =========================================================

BIN_NAME_MAP = {0: "low", 1: "mid", 2: "high"}
SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]

LABEL_ORDER = ["low", "mid", "high"]
SOURCE_ORDER = ["Real", "Generated"]  # GC / k-mer 图继续使用
STRENGTH_SOURCE_ORDER = ["Real", "Pred(Real)", "Pred(Generated)"] # 强度图专用

# =========================================================
# 基础函数
# =========================================================

def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ATCG"])

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_full_limits(series, pad_ratio=0.05, min_pad=0.01):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return None
    lo, hi = np.min(vals), np.max(vals)
    if hi > lo:
        pad = max((hi - lo) * pad_ratio, min_pad)
    else:
        pad = min_pad
    return lo - pad, hi + pad

def load_bin_edges_from_pkl(pkl_path: str):
    """
    对当前 E. coli 流程来说，pkl 可能没有 bin_edges。
    有就读，没有就返回 None。
    """
    if not os.path.exists(pkl_path):
        print(f"[Warn] pkl 不存在: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        print(f"[Info] pkl keys: {list(data.keys())}")
    else:
        print(f"[Warn] pkl 不是 dict，type={type(data)}")
        return None

    if "bin_edges" in data:
        edges = np.asarray(data["bin_edges"], dtype=np.float32)
        print(f"Loaded bin edges from pkl: {edges}")
        return edges

    print(f"[Warn] {pkl_path} 中没有 'bin_edges' 字段，返回 None。")
    return None

def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")

def read_txt_sequences(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到采样序列文件: {path}")

    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if len(seq) > 0:
                seqs.append(seq)

    if len(seqs) == 0:
        raise ValueError(f"文件中没有有效序列: {path}")

    return seqs

# =========================================================
# 读取生成序列
# =========================================================

def load_generated_txts(low_txt, mid_txt, high_txt):
    rows = []

    for seq in read_txt_sequences(low_txt):
        rows.append({"group": "opt_low", "sequence_std": seq})

    for seq in read_txt_sequences(mid_txt):
        rows.append({"group": "opt_mid", "sequence_std": seq})

    for seq in read_txt_sequences(high_txt):
        rows.append({"group": "opt_high", "sequence_std": seq})

    df = pd.DataFrame(rows)
    print("Generated sequences loaded:")
    print(df["group"].value_counts())
    return df

# =========================================================
# 读取真实数据
# 1) 优先直接使用 label
# 2) 如果没有 label，再尝试用 strength + edges
# =========================================================

def load_real_strength_df(path: str, edges=None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到真实数据文件: {path}")

    real_df = pd.read_csv(path)
    real_df.columns = [c.strip().lower() for c in real_df.columns]

    seq_col = find_seq_col(real_df)

    real_df = real_df.copy()
    real_df["sequence_std"] = real_df[seq_col].astype(str).apply(standardize_seq)
    real_df = real_df[real_df["sequence_std"].str.len() > 0].copy()

    if "strength" in real_df.columns:
        real_df["strength"] = pd.to_numeric(real_df["strength"], errors="coerce")

    # ---------- 情况1：CSV 自带 label ----------
    if "label" in real_df.columns:
        real_df["label"] = real_df["label"].astype(str).str.strip().str.lower()
        real_df = real_df[real_df["label"].isin(LABEL_ORDER)].copy()

        if "strength" in real_df.columns:
            real_df = real_df.dropna(subset=["strength"]).copy()

        real_df["group"] = real_df["label"].map(lambda x: f"real_{x}")
        return real_df.reset_index(drop=True)

    # ---------- 情况2：没有 label，需要按 strength + edges 分箱 ----------
    if "strength" not in real_df.columns:
        raise ValueError("真实数据中既没有 label 列，也没有 strength 列，无法构建 real_low/mid/high。")

    real_df["strength"] = pd.to_numeric(real_df["strength"], errors="coerce")
    real_df = real_df.dropna(subset=["strength"]).reset_index(drop=True)

    if edges is None:
        raise ValueError(
            "真实数据没有 label 列，且分箱边缘计算失败。"
            "请检查输入的全集文件。"
        )

    edges = np.asarray(edges, dtype=np.float32)

    real_df["bin_id"] = pd.cut(
        real_df["strength"],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True
    )
    real_df = real_df.dropna(subset=["bin_id"]).copy()
    real_df["bin_id"] = real_df["bin_id"].astype(int)
    real_df["group"] = real_df["bin_id"].map(BIN_NAME_MAP).map(lambda x: f"real_{x}")

    return real_df.reset_index(drop=True)

# =========================================================
# Predictor 预测部分
# =========================================================

def predict_sequences_with_model(sequences, predictor_model_path):
    import re
    import torch

    bio_root = "/home/yt/Code/DNA-Diffusion"
    if bio_root not in sys.path:
        sys.path.insert(0, bio_root)

    from Predictor.predictor_models import LSTMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(predictor_model_path):
        raise FileNotFoundError(f"predictor 模型不存在: {predictor_model_path}")
        
    scaler_pkl_path = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/scaler_EC.pkl"
    if not os.path.exists(scaler_pkl_path):
        raise FileNotFoundError(f"未找到训练时保存的 scaler 文件: {scaler_pkl_path}")
        
    with open(scaler_pkl_path, "rb") as sf:
        scaler = pickle.load(sf)
    print(f"[Info] 成功加载训练期 Scaler: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")

    hidden_size = 256
    dropout_rate = 0.2
    lambda_l2 = 0.001

    model = LSTMModel(4, hidden_size, 1, dropout_rate, lambda_l2)

    state = torch.load(predictor_model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    seq_len = 165

    def string_to_array(my_string: str):
        my_string = str(my_string).lower().replace("u", "t")
        my_string = re.sub("[^acgt]", "z", my_string)
        return np.array(list(my_string))

    def one_hot_encode(my_array):
        mapping = {"a": 0, "c": 1, "g": 2, "t": 3}
        onehot_encoded = np.zeros((len(my_array), 4), dtype=np.float32)
        for i, char in enumerate(my_array):
            if char in mapping:
                onehot_encoded[i, mapping[char]] = 1.0
        return onehot_encoded

    arrs = []
    for s in sequences:
        x = one_hot_encode(string_to_array(s))
        if x.shape[0] < seq_len:
            pad_len = seq_len - x.shape[0]
            x = np.pad(x, ((0, pad_len), (0, 0)), mode="constant")
        elif x.shape[0] > seq_len:
            x = x[:seq_len]
        arrs.append(torch.tensor(x, dtype=torch.float32))

    x = torch.stack(arrs, dim=0).to(device)

    with torch.no_grad():
        y_norm = model(x).detach().cpu().numpy().reshape(-1, 1)
        y_raw = scaler.inverse_transform(y_norm).reshape(-1)

    del model, x, arrs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_raw

def attach_generated_predictions(gen_df, predictor_model_path):
    seqs = gen_df["sequence_std"].tolist()
    preds = predict_sequences_with_model(seqs, predictor_model_path)

    gen_df = gen_df.copy()
    gen_df["pred_strength"] = pd.to_numeric(preds, errors="coerce")
    gen_df = gen_df.dropna(subset=["pred_strength"]).reset_index(drop=True)

    gen_df.to_csv(OUT_GENERATED_PRED_CSV, index=False)
    print(f"Saved generated prediction table to: {OUT_GENERATED_PRED_CSV}")
    return gen_df

def attach_real_predictions(real_df, predictor_model_path, out_csv=None):
    """
    对真实序列做预测，结果写入 pred_strength 列。
    out_csv 为可选保存路径，None 则不保存。
    """
    seqs = real_df["sequence_std"].tolist()
    preds = predict_sequences_with_model(seqs, predictor_model_path)

    real_df = real_df.copy()
    real_df["pred_strength"] = pd.to_numeric(preds, errors="coerce")
    real_df = real_df.dropna(subset=["pred_strength"]).reset_index(drop=True)

    if out_csv is not None:
        real_df.to_csv(out_csv, index=False)
        print(f"Saved real prediction table to: {out_csv}")
    return real_df

# =========================================================
# 强度对比（三档子图箱线/小提琴图）
# =========================================================

def build_strength_plot_df(gen_df, real_test_df):
    real_actual = real_test_df[["group", "strength"]].copy()
    real_actual["source"] = "Real"
    real_actual["bin_level"] = real_actual["group"].str.replace("real_", "", regex=False)
    real_actual = real_actual.rename(columns={"strength": "value"})

    gen_pred = gen_df[["group", "pred_strength"]].copy()
    gen_pred["source"] = "Pred(Generated)"
    gen_pred["bin_level"] = gen_pred["group"].str.replace("opt_", "", regex=False)
    gen_pred = gen_pred.rename(columns={"pred_strength": "value"})

    real_pred = real_test_df[["group", "pred_strength"]].copy()
    real_pred["source"] = "Pred(Real)"
    real_pred["bin_level"] = real_pred["group"].str.replace("real_", "", regex=False)
    real_pred = real_pred.rename(columns={"pred_strength": "value"})

    plot_df = pd.concat([real_actual, gen_pred, real_pred], ignore_index=True)
    plot_df["bin_level"] = pd.Categorical(plot_df["bin_level"], categories=LABEL_ORDER, ordered=True)
    plot_df["source"] = pd.Categorical(plot_df["source"], categories=STRENGTH_SOURCE_ORDER, ordered=True)
    return plot_df

def summarize(plot_df):
    rows = []
    grouped = plot_df.groupby(["bin_level", "source"], observed=True)["value"]
    for (bin_level, source), vals in grouped:
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "bin_level": bin_level,
            "source": source,
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
    color_map = {
        "Real":            "#4D4D4D",
        "Pred(Real)":      "#0072B2",
        "Pred(Generated)": "#D55E00",
    }
    label_map = {
        "Real":            "Real Strength",
        "Pred(Real)":      "Predicted Real",
        "Pred(Generated)": "Predicted Generated",
    }
    pos_map = {
        "low":  {"Real": 0.72, "Pred(Real)": 1.00, "Pred(Generated)": 1.28},
        "mid":  {"Real": 1.72, "Pred(Real)": 2.00, "Pred(Generated)": 2.28},
        "high": {"Real": 2.72, "Pred(Real)": 3.00, "Pred(Generated)": 3.28},
    }

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.4))

        for b in LABEL_ORDER:
            for s in STRENGTH_SOURCE_ORDER:
                vals = plot_df[
                    (plot_df["bin_level"] == b) & (plot_df["source"] == s)
                ]["value"].dropna().values
                if len(vals) == 0:
                    continue
                pos = pos_map[b][s]
                c   = color_map[s]
                ax.boxplot(
                    vals, positions=[pos], widths=0.22,
                    patch_artist=True, showfliers=False,
                    boxprops    =dict(facecolor=c, alpha=0.30, edgecolor=c, linewidth=1.2),
                    medianprops =dict(color=c, linewidth=1.5),
                    whiskerprops=dict(color=c, linewidth=0.8),
                    capprops    =dict(color=c, linewidth=0.8),
                )
                n_show = min(len(vals), 60)
                if n_show:
                    idx = np.random.choice(len(vals), n_show, replace=False)
                    xj  = np.random.normal(pos, 0.03, n_show)
                    ax.scatter(xj, vals[idx], alpha=0.22, s=5,
                               color=c, edgecolors="none", zorder=3)

        handles = [
            Patch(facecolor=color_map[s], edgecolor=color_map[s],
                  alpha=0.55, label=label_map[s])
            for s in STRENGTH_SOURCE_ORDER
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  handlelength=1.2, labelspacing=0.3)

        ax.set_xlim(0.38, 3.62)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(LABEL_ORDER)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Expression Strength")
        ax.set_title("Expression Strength Distribution", pad=6)
        lims = get_full_limits(plot_df["value"], pad_ratio=0.08, min_pad=0.02)
        if lims:
            ax.set_ylim(lims[0], lims[1] + (lims[1] - lims[0]) * 0.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

# =========================================================
# GC 含量
# =========================================================

def gc_content(seq: str) -> float:
    seq = standardize_seq(seq)
    if len(seq) == 0:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)

def build_gc_plot_df(gen_df, real_df):
    real_gc = real_df[["group", "sequence_std"]].copy()
    real_gc["source"] = "Real"
    real_gc["bin_level"] = real_gc["group"].str.replace("real_", "", regex=False)
    real_gc["gc_content"] = real_gc["sequence_std"].apply(gc_content)

    gen_gc = gen_df[["group", "sequence_std"]].copy()
    gen_gc["source"] = "Generated"
    gen_gc["bin_level"] = gen_gc["group"].str.replace("opt_", "", regex=False)
    gen_gc["gc_content"] = gen_gc["sequence_std"].apply(gc_content)

    gc_df = pd.concat([real_gc, gen_gc], ignore_index=True).dropna(subset=["gc_content"])
    gc_df["bin_level"] = pd.Categorical(gc_df["bin_level"], categories=LABEL_ORDER, ordered=True)
    gc_df["source"] = pd.Categorical(gc_df["source"], categories=SOURCE_ORDER, ordered=True)
    return gc_df

def plot_gc_box(gc_df, out_path):
    color_map = {"Real": "#0072B2", "Generated": "#D55E00"}
    pos_map = {
        "low":  {"Real": 0.85, "Generated": 1.15},
        "mid":  {"Real": 1.85, "Generated": 2.15},
        "high": {"Real": 2.85, "Generated": 3.15},
    }

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.4))

        for b in LABEL_ORDER:
            for s in SOURCE_ORDER:
                vals = gc_df[
                    (gc_df["bin_level"] == b) & (gc_df["source"] == s)
                ]["gc_content"].dropna().values
                if len(vals) == 0:
                    continue
                pos = pos_map[b][s]
                c   = color_map[s]
                ax.boxplot(
                    vals, positions=[pos], widths=0.22,
                    patch_artist=True, showfliers=False,
                    boxprops    =dict(facecolor=c, alpha=0.30, edgecolor=c, linewidth=1.2),
                    medianprops =dict(color=c, linewidth=1.5),
                    whiskerprops=dict(color=c, linewidth=0.8),
                    capprops    =dict(color=c, linewidth=0.8),
                )
                n_show = min(len(vals), 80)
                if n_show:
                    idx = np.random.choice(len(vals), n_show, replace=False)
                    xj  = np.random.normal(pos, 0.024, n_show)
                    ax.scatter(xj, vals[idx], alpha=0.20, s=6,
                               color=c, edgecolors="none", zorder=3)

        handles = [
            Patch(facecolor=color_map["Real"],      edgecolor=color_map["Real"],      alpha=0.55, label="Real"),
            Patch(facecolor=color_map["Generated"], edgecolor=color_map["Generated"], alpha=0.55, label="Generated"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  handlelength=1.2, labelspacing=0.3)

        ax.set_xlim(0.62, 3.38)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(LABEL_ORDER)
        ax.set_xlabel("Bin")
        ax.set_ylabel("GC Content")
        ax.set_title("GC Content Consistency", pad=6)
        lims = get_full_limits(gc_df["gc_content"], pad_ratio=0.06, min_pad=0.015)
        if lims:
            ax.set_ylim(*lims)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

# =========================================================
# k-mer PCC
# =========================================================

def all_kmers(k):
    return ["".join(p) for p in product("ATCG", repeat=k)]

def kmer_freq_vector(seqs, k):
    vocab = all_kmers(k)
    counter = Counter()
    total = 0

    for seq in seqs:
        seq = standardize_seq(seq)
        if len(seq) < k:
            continue
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if set(kmer).issubset(set("ATCG")):
                counter[kmer] += 1
                total += 1

    if total == 0:
        return np.zeros(len(vocab))

    return np.array([counter[km] / total for km in vocab])

def safe_pearson(x, y):
    return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan

def compute_kmer_pcc(real_df, gen_df):
    rows = []
    for bin_name in LABEL_ORDER:
        real_seqs = real_df.loc[real_df["group"] == f"real_{bin_name}", "sequence_std"].tolist()
        gen_seqs = gen_df.loc[gen_df["group"] == f"opt_{bin_name}", "sequence_std"].tolist()

        for k in [3, 4, 5]:
            pcc = safe_pearson(
                kmer_freq_vector(real_seqs, k),
                kmer_freq_vector(gen_seqs, k)
            )
            rows.append({
                "condition": bin_name,
                "kmer": f"{k}-mer",
                "pcc": pcc
            })
    return pd.DataFrame(rows)

def plot_kmer_pcc_bar(kmer_df, out_path):
    colors  = {"3-mer": "#999999", "4-mer": "#0072B2", "5-mer": "#D55E00"}
    kmers   = ["3-mer", "4-mer", "5-mer"]
    conds   = LABEL_ORDER
    x       = np.arange(len(conds))
    width   = 0.22
    offsets = [-width, 0, width]

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.4))

        all_vals = []
        for i, k in enumerate(kmers):
            vals = []
            for c in conds:
                sub = kmer_df[(kmer_df["condition"] == c) & (kmer_df["kmer"] == k)]
                vals.append(float(sub["pcc"].values[0]) if len(sub) else np.nan)
            all_vals.extend([v for v in vals if not pd.isna(v)])
            bars = ax.bar(x + offsets[i], vals, width=width, label=k,
                          color=colors[k], alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not pd.isna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.003, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=6.5, color="#333333")

        ymax = max(all_vals) if all_vals else 1.0
        ax.set_ylim(0, min(1.06, ymax + 0.10))
        ax.set_xticks(x)
        ax.set_xticklabels(conds)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Pearson Correlation")
        ax.set_title("3/4/5-mer Frequency Correlation", pad=6)
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.1, labelspacing=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

# ================== 真实 vs 生成序列 Oracle 预测相关性散点图 ==================
def plot_oracle_scatter(real_df, gen_df, out_dir):
    import scipy.stats as stats

    real_exp   = real_df["strength"].values
    real_pred  = real_df["oracle_pred_strength"].values

    mask = ~(np.isnan(real_exp) | np.isnan(real_pred))
    real_exp, real_pred = real_exp[mask], real_pred[mask]

    r, p = stats.pearsonr(real_exp, real_pred)

    out_path = os.path.join(out_dir, "oracle_scatter_real_exp_vs_pred.png")

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.4))

        color_map = {"low": "#4393C3", "mid": "#74C476", "high": "#D6604D"}
        for lbl in LABEL_ORDER:
            # 兼容两种情况：CSV 自带 label 列 或 由 group 列推导
            if "label" in real_df.columns:
                sub = real_df[real_df["label"] == lbl]
            else:
                sub = real_df[real_df["group"] == f"real_{lbl}"]
            x = sub["strength"].values
            y = sub["oracle_pred_strength"].values
            mask2 = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(x[mask2], y[mask2],
                       color=color_map[lbl], alpha=0.45, s=14,
                       edgecolors="none", label=lbl, zorder=3)

        m, b = np.polyfit(real_exp, real_pred, 1)
        xline = np.linspace(real_exp.min(), real_exp.max(), 200)
        ax.plot(xline, m * xline + b, color="#333333", linewidth=1.2, zorder=4)

        lim_lo = min(real_exp.min(), real_pred.min())
        lim_hi = max(real_exp.max(), real_pred.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)

        p_str = f"p = {p:.2e}" if p >= 1e-4 else f"p < 1e-4"
        ax.text(0.05, 0.93, f"r = {r:.3f}\n{p_str}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", color="#222222")

        ax.set_xlabel("Experimental Strength (Real)")
        ax.set_ylabel("Oracle Predicted Strength")
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.0, labelspacing=0.3, markerscale=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"散点图已保存: {out_path}")


def plot_generated_oracle_scatter(real_df, gen_df, out_dir):
    """生成序列 Q-Q 图，图例内标注各 bin 的 5-mer PCC，不再显示整体 r 值。"""

    real_sorted = np.sort(real_df["oracle_pred_strength"].dropna().values)

    # 计算各 bin 的 5-mer PCC
    pcc_labels = {}
    for bin_name in LABEL_ORDER:
        real_seqs = real_df.loc[real_df["group"] == f"real_{bin_name}", "sequence_std"].tolist()
        gen_seqs  = gen_df.loc[gen_df["group"]  == f"opt_{bin_name}",  "sequence_std"].tolist()
        pcc_labels[bin_name] = safe_pearson(
            kmer_freq_vector(real_seqs, 5),
            kmer_freq_vector(gen_seqs,  5)
        )

    out_path  = os.path.join(out_dir, "oracle_scatter_real_vs_generated_qq.png")
    color_map = {"opt_low": "#4393C3", "opt_mid": "#74C476", "opt_high": "#D6604D"}
    label_map = {"opt_low": "low",     "opt_mid": "mid",     "opt_high": "high"}

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.4))

        all_rq, all_gq = [], []
        for grp in ["opt_low", "opt_mid", "opt_high"]:
            g_vals = np.sort(gen_df.loc[gen_df["group"] == grp,
                                        "pred_strength"].dropna().values)
            if len(g_vals) == 0:
                continue
            r_q      = np.quantile(real_sorted, np.linspace(0, 1, len(g_vals)))
            bin_name = label_map[grp]
            pcc_val  = pcc_labels.get(bin_name, np.nan)
            leg_lbl  = (f"{bin_name}  (5-mer r={pcc_val:.3f})"
                        if not np.isnan(pcc_val) else bin_name)
            ax.scatter(r_q, g_vals, color=color_map[grp], alpha=0.40, s=12,
                       edgecolors="none", label=leg_lbl, zorder=3)
            all_rq.extend(r_q.tolist())
            all_gq.extend(g_vals.tolist())

        all_rq = np.array(all_rq)
        all_gq = np.array(all_gq)
        if len(all_rq) > 1:
            m, b  = np.polyfit(all_rq, all_gq, 1)
            xline = np.linspace(all_rq.min(), all_rq.max(), 200)
            ax.plot(xline, m * xline + b, color="#333333", linewidth=1.2, zorder=4)

        lim_lo = min(all_rq.min(), all_gq.min())
        lim_hi = max(all_rq.max(), all_gq.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)

        ax.set_xlabel("Real Oracle Predicted Strength")
        ax.set_ylabel("Generated Oracle Predicted Strength")
        # ax.set_title("Q-Q: Real vs Generated", pad=6)
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.0, labelspacing=0.35, markerscale=1.2, fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"散点图已保存: {out_path}")

# ================== 5-mer 全局散点图（所有 bin 汇总） ==================
def plot_kmer5_global_scatter(real_df, gen_df, out_dir):
    import scipy.stats as stats

    real_seqs = real_df["sequence_std"].tolist()
    gen_seqs  = gen_df["sequence_std"].tolist()

    real_vec = kmer_freq_vector(real_seqs, 5)
    gen_vec  = kmer_freq_vector(gen_seqs,  5)

    r, p = stats.pearsonr(real_vec, gen_vec)
    out_path = os.path.join(out_dir, "kmer5_global_scatter.png")

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.4))

        ax.scatter(real_vec, gen_vec,
                   color="#0072B2", alpha=0.35, s=8, edgecolors="none",
                   label=f"5-mer (n=1024)", zorder=3)

        # fit line
        m, b   = np.polyfit(real_vec, gen_vec, 1)
        xline  = np.linspace(real_vec.min(), real_vec.max(), 300)
        ax.plot(xline, m * xline + b, color="#005691", linewidth=1.2, zorder=4)

        # y=x reference
        lo = min(real_vec.min(), gen_vec.min())
        hi = max(real_vec.max(), gen_vec.max())
        ax.plot([lo, hi], [lo, hi],
                color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)

        p_str = f"p = {p:.2e}" if p >= 1e-4 else "p < 1e-4"
        ax.text(0.05, 0.93, f"r = {r:.3f}\n{p_str}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", color="#222222")

        ax.set_xlabel("Real 5-mer frequency")
        ax.set_ylabel("Generated 5-mer frequency")
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.0, labelspacing=0.3, markerscale=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"5-mer 全局散点图已保存: {out_path}")
    return r, p

# =========================================================
# 主函数
# =========================================================

def main():
    ensure_dir(OUT_DIR)

    # 1. 首先尝试从原作者配置的 pkl 中加载边界
    edges = load_bin_edges_from_pkl(PKL_PATH)
    
    # 2. 如果 pkl 缺失边界或无法读取，开启自动等频自适应分箱机制
    if edges is None:
        if TEST_BIN_EDGES_OVERRIDE is not None:
            edges = np.asarray(TEST_BIN_EDGES_OVERRIDE, dtype=np.float32)
            print(f"[Info] 使用全局指定边界线: {edges}")
        else:
            print("[Info] 正在根据 REAL_CSV 的表达强度自动计算当前数据集的等频三分边界线...")
            if not os.path.exists(REAL_CSV):
                raise FileNotFoundError(f"自适应失败：找不到训练全集数据文件 {REAL_CSV}")
            
            tmp_df = pd.read_csv(REAL_CSV)
            tmp_df.columns = [c.strip().lower() for c in tmp_df.columns]
            if "strength" not in tmp_df.columns:
                raise ValueError(f"无法自动算边界：全集 CSV ({REAL_CSV}) 缺乏 'strength' 表达强度列。")
                
            strengths = pd.to_numeric(tmp_df["strength"], errors="coerce").dropna().values
            q1 = np.percentile(strengths, 33.333)
            q2 = np.percentile(strengths, 66.667)
            
            edges = np.array([-np.inf, q1, q2, np.inf], dtype=np.float32)
            print(f"[Success] 自适应分箱边界确定成功: {edges}")
            print(f"          分箱标准: Low <= {q1:.4f} < Mid <= {q2:.4f} < High")

    # ── 读取并组合生成的文本序列 ──────────────────────────────────────
    gen_df = load_generated_txts(GEN_LOW_TXT, GEN_MID_TXT, GEN_HIGH_TXT)

    # ── 预测生成序列的表达强度 ─────────────────────────────────────────
    gen_df = attach_generated_predictions(gen_df, PREDICTOR_MODEL_PATH)

    # ── 加载训练全集（GC / k-mer 特征比对 + oracle 散点图） ────────────
    real_df = load_real_strength_df(REAL_CSV, edges)

    # ✅ 修复：对全集 real_df 做预测，结果列命名为 oracle_pred_strength
    #    使用独立的保存路径，避免覆盖测试集预测文件
    real_df = attach_real_predictions(real_df, PREDICTOR_MODEL_PATH,
                                      out_csv=OUT_REAL_ALL_PRED_CSV)
    real_df = real_df.rename(columns={"pred_strength": "oracle_pred_strength"})

    # ── 加载独立测试集并做模型强度预测（强度分布箱线图专用） ──────────
    real_test_df = load_real_strength_df(TEST_REAL_CSV, edges)
    real_test_df = attach_real_predictions(real_test_df, PREDICTOR_MODEL_PATH,
                                           out_csv=OUT_REAL_TEST_PRED_CSV)

    # ── 打印预测器在独立测试集上的泛化精度 ───────────────────────────
    from scipy.stats import pearsonr
    r, p = pearsonr(real_test_df["strength"], real_test_df["pred_strength"])
    print(f"[Validation] Predictor 泛化指标 (Test Set): Pearson R = {r:.3f}, R² = {r**2:.3f}")

    # ================== 1. 强度对比分布（3分档图） ==================
    strength_plot_df = build_strength_plot_df(gen_df, real_test_df)
    summarize(strength_plot_df).to_csv(OUT_COMPARE_CSV, index=False)
    plot_strength_box(strength_plot_df, OUT_COMPARE_PLOT)
    print(f"强度箱线图已保存在: {OUT_COMPARE_PLOT}")

    # ================== 2. GC 含量一致性对比 ==================
    gc_df = build_gc_plot_df(gen_df, real_df)
    gc_df.groupby(["bin_level", "source"], observed=True)["gc_content"] \
        .agg(["count", "mean", "std", "min", "median", "max"]) \
        .reset_index() \
        .to_csv(OUT_GC_CSV, index=False)
    plot_gc_box(gc_df, OUT_GC_PLOT)
    print(f"GC含量箱线图已保存在: {OUT_GC_PLOT}")

    # ================== 3. k-mer 皮尔逊相关分析 ==================
    kmer_df = compute_kmer_pcc(real_df, gen_df)
    kmer_df.to_csv(OUT_KMER_PCC_CSV, index=False)
    plot_kmer_pcc_bar(kmer_df, OUT_KMER_PLOT)
    print(f"k-mer相关柱状图已保存在: {OUT_KMER_PLOT}")

    # ================== 4. Oracle 散点图 ==================
    # ✅ 修复：real_df 已具备 oracle_pred_strength 列，可正常执行
    r, p = pearsonr(real_df["strength"], real_df["oracle_pred_strength"])
    print(f"[Oracle] 全集预测相关性: Pearson R = {r:.3f}, R² = {r**2:.3f}")

    plot_oracle_scatter(real_df, gen_df, OUT_DIR)
    plot_generated_oracle_scatter(real_df, gen_df, OUT_DIR)
    # ================== 5. 5-mer 全局散点图 ==================
    plot_kmer5_global_scatter(real_df, gen_df, OUT_DIR)


if __name__ == "__main__":
    main()