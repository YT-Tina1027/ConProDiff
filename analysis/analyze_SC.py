import os
import sys
import gc
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product
from matplotlib.patches import Patch
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
# ================== 静音设置 ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ================== 全局绘图风格（Nature 论文级） ==================
plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             14,
    "axes.labelsize":        12,
    "axes.titlesize":        12,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "legend.fontsize":       12,
    "legend.title_fontsize": 12,
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
VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# ================== 路径配置（请按需修改） ==================
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/yeast_gxlx/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/yeast_gxlx/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/yeast_gxlx/high.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"

ORACLE_DIR = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_MODEL_CONDITIONS = "defined_media"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/yeast_gclx_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# 输出文件
OUT_GENERATED_PRED_CSV = os.path.join(OUT_DIR, "generated_with_oracle_pred_strength.csv")
OUT_REAL_PRED_CSV      = os.path.join(OUT_DIR, "real_with_oracle_pred_strength.csv")

OUT_COMPARE_CSV  = os.path.join(OUT_DIR, "strength_three_source_summary.csv")
OUT_COMPARE_PLOT = os.path.join(OUT_DIR, "strength_three_source_box.png")

OUT_GC_CSV   = os.path.join(OUT_DIR, "gc_content_summary.csv")
OUT_GC_PLOT  = os.path.join(OUT_DIR, "gc_content_box.png")

OUT_KMER_PCC_CSV = os.path.join(OUT_DIR, "kmer_345_pcc_summary.csv")
OUT_KMER_PLOT    = os.path.join(OUT_DIR, "kmer_345_pcc_bar.png")

SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]
LABEL_ORDER = ["low", "mid", "high"]

# 强度图三组来源
STRENGTH_SOURCE_ORDER = ["Real-Exp", "Real-Oracle", "Generated-Oracle"]
# GC / kmer 图两组来源
PAIR_SOURCE_ORDER = ["Real", "Generated"]


# ================== 辅助函数 ==================
def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ATCG"])


def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")


def read_txt_sequences(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件: {path}")
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if seq:
                seqs.append(seq)
    if not seqs:
        raise ValueError(f"文件中无有效序列: {path}")
    return seqs


def get_full_limits(series, pad_ratio=0.05, min_pad=0.01):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return None
    lo, hi = np.min(vals), np.max(vals)
    pad = max((hi - lo) * pad_ratio, min_pad) if hi > lo else min_pad
    return lo - pad, hi + pad


# ================== 数据加载 ==================
def load_generated_txts(low_txt, mid_txt, high_txt):
    rows = []
    for seq in read_txt_sequences(low_txt):
        rows.append({"group": "opt_low", "sequence_std": seq})
    for seq in read_txt_sequences(mid_txt):
        rows.append({"group": "opt_mid", "sequence_std": seq})
    for seq in read_txt_sequences(high_txt):
        rows.append({"group": "opt_high", "sequence_std": seq})
    df = pd.DataFrame(rows)
    vprint("Generated sequences loaded:\n", df["group"].value_counts())
    return df


def load_real_strength_df_from_label(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到真实数据文件: {path}")

    real_df = pd.read_csv(path)
    real_df.columns = [c.strip().lower() for c in real_df.columns]

    if "strength" not in real_df.columns:
        strength_candidates = ["activity", "score", "expression", "expr", "value"]
        found_col = next((c for c in strength_candidates if c in real_df.columns), None)
        if found_col is None:
            raise ValueError(
                f"真实数据中缺少 strength 列。当前列为: {list(real_df.columns)}"
            )
        print(f"[Info] 未找到 strength 列，使用 {found_col} 作为 strength。")
        real_df["strength"] = real_df[found_col]

    seq_col = find_seq_col(real_df)
    real_df = real_df.copy()
    real_df["strength"] = pd.to_numeric(real_df["strength"], errors="coerce")
    real_df["sequence_std"] = real_df[seq_col].astype(str).apply(standardize_seq)
    real_df = real_df.dropna(subset=["strength", "sequence_std"]).reset_index(drop=True)
    real_df = real_df[real_df["sequence_std"].str.len() > 0].copy()

    if len(real_df) == 0:
        raise ValueError("真实数据清洗后为空，请检查 sequence 和 strength 列。")

    if "label" not in real_df.columns:
        print("[Info] 真实数据中没有 label 列，使用 strength 三分位自动划分 low/mid/high。")
        q1 = real_df["strength"].quantile(1.0 / 3.0)
        q2 = real_df["strength"].quantile(2.0 / 3.0)
        print(f"[Info] low: <={q1:.6f}, mid: <={q2:.6f}, high: >{q2:.6f}")

        def assign_quantile_label(x):
            if x <= q1:   return "low"
            elif x <= q2: return "mid"
            else:         return "high"

        real_df["label"] = real_df["strength"].apply(assign_quantile_label)
    else:
        print("[Info] 检测到已有 label 列，直接使用原始 label。")
        real_df["label"] = real_df["label"].astype(str).str.strip().str.lower()

    real_df = real_df[real_df["label"].isin(LABEL_ORDER)].copy()
    if len(real_df) == 0:
        raise ValueError("真实数据经过 label 筛选后为空。")

    real_df["group"] = real_df["label"].map(lambda x: f"real_{x}")

    print("\n[Info] Real data label counts:")
    print(real_df["label"].value_counts().reindex(LABEL_ORDER))
    print("\n[Info] Real data strength summary by label:")
    print(real_df.groupby("label")["strength"]
          .agg(["count", "mean", "std", "min", "median", "max"])
          .reindex(LABEL_ORDER))

    return real_df.reset_index(drop=True)


# ================== Oracle 预测 ==================
def predict_sequences_with_oracle(sequences, oracle_dir, model_conditions):
    import tensorflow as tf

    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)

    from aux import load_model, evaluate_model

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    fitness_function_graph = tf.Graph()
    with fitness_function_graph.as_default():
        model, scaler, batch_size = load_model(model_conditions)

    preds = evaluate_model(sequences, model, scaler, batch_size, fitness_function_graph)
    return np.asarray(preds, dtype=float).reshape(-1)


def attach_generated_predictions(gen_df, oracle_dir, model_conditions):
    seqs  = gen_df["sequence_std"].tolist()
    preds = predict_sequences_with_oracle(seqs, oracle_dir, model_conditions)
    gen_df = gen_df.copy()
    gen_df["pred_strength"] = pd.to_numeric(preds, errors="coerce")
    gen_df = gen_df.dropna(subset=["pred_strength"]).reset_index(drop=True)
    gen_df.to_csv(OUT_GENERATED_PRED_CSV, index=False)
    return gen_df


def attach_real_oracle_predictions(real_df, oracle_dir, model_conditions):
    seqs  = real_df["sequence_std"].tolist()
    preds = predict_sequences_with_oracle(seqs, oracle_dir, model_conditions)
    real_df = real_df.copy()
    real_df["oracle_pred_strength"] = pd.to_numeric(preds, errors="coerce")
    real_df = real_df.dropna(subset=["oracle_pred_strength"]).reset_index(drop=True)
    real_df.to_csv(OUT_REAL_PRED_CSV, index=False)
    return real_df


# ================== 强度对比：三组来源 ==================
def build_strength_plot_df(gen_df, real_df):
    real_exp = real_df[["group", "strength"]].copy()
    real_exp["source"]    = "Real-Exp"
    real_exp["bin_level"] = real_exp["group"].str.replace("real_", "", regex=False)
    real_exp = real_exp.rename(columns={"strength": "value"})

    real_oracle = real_df[["group", "oracle_pred_strength"]].copy()
    real_oracle["source"]    = "Real-Oracle"
    real_oracle["bin_level"] = real_oracle["group"].str.replace("real_", "", regex=False)
    real_oracle = real_oracle.rename(columns={"oracle_pred_strength": "value"})

    gen_oracle = gen_df[["group", "pred_strength"]].copy()
    gen_oracle["source"]    = "Generated-Oracle"
    gen_oracle["bin_level"] = gen_oracle["group"].str.replace("opt_", "", regex=False)
    gen_oracle = gen_oracle.rename(columns={"pred_strength": "value"})

    plot_df = pd.concat([real_exp, real_oracle, gen_oracle], ignore_index=True)
    plot_df["bin_level"] = pd.Categorical(plot_df["bin_level"], categories=LABEL_ORDER, ordered=True)
    plot_df["source"]    = pd.Categorical(plot_df["source"], categories=STRENGTH_SOURCE_ORDER, ordered=True)
    return plot_df


def summarize_strength(plot_df):
    rows = []
    for (bin_level, source), vals in plot_df.groupby(["bin_level", "source"], observed=True)["value"]:
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "bin_level": bin_level, "source": source,
            "count": len(vals), "mean": vals.mean(), "std": vals.std(),
            "min": vals.min(), "q25": vals.quantile(0.25),
            "median": vals.median(), "q75": vals.quantile(0.75), "max": vals.max()
        })
    return pd.DataFrame(rows)


def plot_strength_box(plot_df, out_path):
    # ✅ 修复：color_map / label_map / pos_map 的 key 与 STRENGTH_SOURCE_ORDER 保持一致
    color_map = {
        "Real-Exp":        "#4D4D4D",
        "Real-Oracle":     "#0072B2",
        "Generated-Oracle":"#D55E00",
    }
    label_map = {
        "Real-Exp":        "Real Strength",
        "Real-Oracle":     "Oracle (Real)",
        "Generated-Oracle":"Oracle (Generated)",
    }
    pos_map = {
        "low":  {"Real-Exp": 0.72, "Real-Oracle": 1.00, "Generated-Oracle": 1.28},
        "mid":  {"Real-Exp": 1.72, "Real-Oracle": 2.00, "Generated-Oracle": 2.28},
        "high": {"Real-Exp": 2.72, "Real-Oracle": 3.00, "Generated-Oracle": 3.28},
    }

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))

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
                    vals, positions=[pos], widths=0.18,
                    patch_artist=True, showfliers=False,
                    boxprops    =dict(facecolor=c, alpha=0.30, edgecolor=c, linewidth=1.2),
                    medianprops =dict(color=c, linewidth=1.5),
                    whiskerprops=dict(color=c, linewidth=0.8),
                    capprops    =dict(color=c, linewidth=0.8),
                )
                n_show = min(len(vals), 60)
                if n_show:
                    idx = np.random.choice(len(vals), n_show, replace=False)
                    xj  = np.random.normal(pos, 0.020, n_show)
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
        # ax.set_xlabel("Bin")
        ax.set_ylabel("Expression Strength")
        lims = get_full_limits(plot_df["value"], pad_ratio=0.08, min_pad=0.02)
        if lims:
            ax.set_ylim(lims[0], lims[1] + (lims[1] - lims[0]) * 0.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


# ================== GC含量分析 ==================
def gc_content(seq: str) -> float:
    seq = standardize_seq(seq)
    if len(seq) == 0:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def build_gc_plot_df(gen_df, real_df):
    real_gc = real_df[["group", "sequence_std"]].copy()
    real_gc["source"]    = "Real"
    real_gc["bin_level"] = real_gc["group"].str.replace("real_", "", regex=False)
    real_gc["gc_content"] = real_gc["sequence_std"].apply(gc_content)

    gen_gc = gen_df[["group", "sequence_std"]].copy()
    gen_gc["source"]    = "Generated"
    gen_gc["bin_level"] = gen_gc["group"].str.replace("opt_", "", regex=False)
    gen_gc["gc_content"] = gen_gc["sequence_std"].apply(gc_content)

    gc_df = pd.concat([real_gc, gen_gc], ignore_index=True).dropna(subset=["gc_content"])
    gc_df["bin_level"] = pd.Categorical(gc_df["bin_level"], categories=LABEL_ORDER, ordered=True)
    gc_df["source"]    = pd.Categorical(gc_df["source"], categories=PAIR_SOURCE_ORDER, ordered=True)
    return gc_df


def plot_gc_box(gc_df, out_path):
    color_map = {"Real": "#0072B2", "Generated": "#D55E00"}
    pos_map = {
        "low":  {"Real": 0.85, "Generated": 1.15},
        "mid":  {"Real": 1.85, "Generated": 2.15},
        "high": {"Real": 2.85, "Generated": 3.15},
    }

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))

        for b in LABEL_ORDER:
            for s in PAIR_SOURCE_ORDER:   # ✅ 修复：原来用了未定义的 SOURCE_ORDER
                vals = gc_df[
                    (gc_df["bin_level"] == b) & (gc_df["source"] == s)
                ]["gc_content"].dropna().values
                if len(vals) == 0:
                    continue
                pos = pos_map[b][s]
                c   = color_map[s]
                ax.boxplot(
                    vals, positions=[pos], widths=0.20,
                    patch_artist=True, showfliers=False,
                    boxprops    =dict(facecolor=c, alpha=0.30, edgecolor=c, linewidth=1.2),
                    medianprops =dict(color=c, linewidth=1.5),
                    whiskerprops=dict(color=c, linewidth=0.8),
                    capprops    =dict(color=c, linewidth=0.8),
                )
                n_show = min(len(vals), 80)
                if n_show:
                    idx = np.random.choice(len(vals), n_show, replace=False)
                    xj  = np.random.normal(pos, 0.022, n_show)
                    ax.scatter(xj, vals[idx], alpha=0.20, s=6,
                               color=c, edgecolors="none", zorder=3)

        handles = [
            Patch(facecolor=color_map["Real"],      edgecolor=color_map["Real"],      alpha=0.55, label="Real"),
            Patch(facecolor=color_map["Generated"], edgecolor=color_map["Generated"], alpha=0.55, label="Generated"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  handlelength=1.2, labelspacing=0.3)

        ax.set_xlim(0.62, 3.38)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(LABEL_ORDER)
        # ax.set_xlabel("Bin")
        ax.set_ylabel("GC Content")
        lims = get_full_limits(gc_df["gc_content"], pad_ratio=0.06, min_pad=0.015)
        if lims:
            ax.set_ylim(*lims)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


# ================== k-mer PCC分析 ==================
def all_kmers(k):
    return ["".join(p) for p in product("ATCG", repeat=k)]


def kmer_freq_vector(seqs, k):
    vocab   = all_kmers(k)
    counter = Counter()
    total   = 0
    for seq in seqs:
        seq = standardize_seq(seq)
        if len(seq) < k:
            continue
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if set(kmer).issubset("ATCG"):
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
        gen_seqs  = gen_df.loc[gen_df["group"]  == f"opt_{bin_name}",  "sequence_std"].tolist()
        for k in [3, 4, 5]:
            pcc = safe_pearson(kmer_freq_vector(real_seqs, k), kmer_freq_vector(gen_seqs, k))
            rows.append({"condition": bin_name, "kmer": f"{k}-mer", "pcc": pcc})
    return pd.DataFrame(rows)


def plot_kmer_pcc_bar(kmer_df, out_path):
    colors  = {"3-mer": "#999999", "4-mer": "#0072B2", "5-mer": "#D55E00"}
    kmers   = ["3-mer", "4-mer", "5-mer"]
    x       = np.arange(len(LABEL_ORDER))
    width   = 0.22
    offsets = [-width, 0, width]

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))

        all_vals = []
        for i, k in enumerate(kmers):
            vals = []
            for c in LABEL_ORDER:
                sub = kmer_df[(kmer_df["condition"] == c) & (kmer_df["kmer"] == k)]
                vals.append(float(sub["pcc"].values[0]) if len(sub) else np.nan)
            all_vals.extend([v for v in vals if not pd.isna(v)])
            bars = ax.bar(x + offsets[i], vals, width=width, label=k,
                          color=colors[k], alpha=0.85, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not pd.isna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.003, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=6.5, color="#333333")

        ymax = max(all_vals) if all_vals else 1.0
        ax.set_ylim(0, min(1.06, ymax + 0.10))
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_ORDER)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Pearson Correlation")
        ax.legend(loc="lower right", frameon=False, handlelength=1.1, labelspacing=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


# ================== 散点图1：真实实验值 vs Oracle 预测值 ==================
def plot_oracle_scatter(real_df, out_dir):
    import scipy.stats as stats

    real_exp  = real_df["strength"].values
    real_pred = real_df["oracle_pred_strength"].values
    mask = ~(np.isnan(real_exp) | np.isnan(real_pred))
    real_exp, real_pred = real_exp[mask], real_pred[mask]

    r, p = stats.pearsonr(real_exp, real_pred)
    out_path = os.path.join(out_dir, "oracle_scatter_real_exp_vs_pred.png")

    color_map = {"low": "#4393C3", "mid": "#74C476", "high": "#D6604D"}

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))

        for lbl in LABEL_ORDER:
            if "label" in real_df.columns:
                sub = real_df[real_df["label"] == lbl]
            else:
                sub = real_df[real_df["group"] == f"real_{lbl}"]
            x = sub["strength"].values
            y = sub["oracle_pred_strength"].values
            mask2 = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(x[mask2], y[mask2], color=color_map[lbl], alpha=0.45, s=14,
                       edgecolors="none", label=lbl, zorder=3)

        m, b = np.polyfit(real_exp, real_pred, 1)
        xline = np.linspace(real_exp.min(), real_exp.max(), 200)
        ax.plot(xline, m * xline + b, color="#333333", linewidth=1.2, zorder=4)

        lim_lo = min(real_exp.min(), real_pred.min())
        lim_hi = max(real_exp.max(), real_pred.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)

        p_str = f"p = {p:.2e}" if p >= 1e-4 else "p < 1e-4"
        fp = FontProperties(size=14)  # 想要多大写多大
        ax.text(0.05, 0.93, f"r = {r:.3f}\n{p_str}",
                transform=ax.transAxes, fontproperties=fp,      
                verticalalignment="top", color="#222222")

        ax.set_xlabel("Experimental Strength (Real)")
        ax.set_ylabel("Oracle Predicted Strength")
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.0, labelspacing=0.3, markerscale=1.2, fontsize=13)
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

    np.random.seed(42)
    if len(real_seqs) > len(gen_seqs):
        real_seqs = list(np.random.choice(real_seqs, len(gen_seqs), replace=False))

    real_vec = kmer_freq_vector(real_seqs, 5)
    gen_vec  = kmer_freq_vector(gen_seqs,  5)

    r, p = stats.pearsonr(real_vec, gen_vec)
    out_path = os.path.join(out_dir, "kmer5_global_scatter.png")

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

        ax.scatter(real_vec, gen_vec,
                   color="#0072B2", alpha=0.35, s=8, edgecolors="none",
                   label=f"5-mer (n=1024)", zorder=3)

        m, b  = np.polyfit(real_vec, gen_vec, 1)
        x_max = max(real_vec.max(), gen_vec.max())

        xline = np.linspace(0, x_max, 300)
        ax.plot(xline, m * xline + b, color="#005691", linewidth=1.2, zorder=4)

        ax.plot([0, x_max], [0, x_max],
                color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)

        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(0, x_max * 1.05)

        p_str = f"p = {p:.2e}" if p >= 1e-4 else "p < 1e-4"
        fp = FontProperties(size=14) 
        ax.text(0.05, 0.93, f"r = {r:.3f}\n{p_str}",
                transform=ax.transAxes, fontproperties=fp,
                verticalalignment="top", color="#222222")

        ax.set_xlabel("Real 5-mer frequency", fontsize=12)
        ax.set_ylabel("Generated 5-mer frequency", fontsize=12)
        ax.tick_params(axis='both', labelsize=11)    
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.0, labelspacing=0.3, markerscale=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"5-mer 全局散点图已保存: {out_path}")
    return r, p

def plot_generated_oracle_scatter(real_df, gen_df, out_dir):
    real_sorted = np.sort(real_df["oracle_pred_strength"].dropna().values)

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
        fig, ax = plt.subplots(figsize=(3.8, 3.6))

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
        ax.legend(loc="lower right", frameon=False,
                  handlelength=1.0, labelspacing=0.35, markerscale=1.2, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"散点图已保存: {out_path}")
# ================== 主函数 ==================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 读取生成序列并做 oracle 预测
    gen_df = load_generated_txts(GEN_LOW_TXT, GEN_MID_TXT, GEN_HIGH_TXT)
    gen_df = attach_generated_predictions(gen_df, ORACLE_DIR, ORACLE_MODEL_CONDITIONS)

    # 2) 读取真实数据，并给真实序列也做 oracle 预测
    real_df = load_real_strength_df_from_label(REAL_CSV)
    real_df = attach_real_oracle_predictions(real_df, ORACLE_DIR, ORACLE_MODEL_CONDITIONS)

    from scipy.stats import pearsonr
    r, p = pearsonr(real_df["strength"], real_df["oracle_pred_strength"])
    print(f"[Oracle 验证] Pearson R = {r:.3f}, R² = {r**2:.3f}")

    # ================== 1. 强度对比（三组） ==================
    strength_plot_df = build_strength_plot_df(gen_df, real_df)
    summarize_strength(strength_plot_df).to_csv(OUT_COMPARE_CSV, index=False)
    plot_strength_box(strength_plot_df, OUT_COMPARE_PLOT)
    print(f"强度图已保存: {OUT_COMPARE_PLOT}")

    # ================== 2. GC含量 ==================
    gc_df = build_gc_plot_df(gen_df, real_df)
    gc_df.groupby(["bin_level", "source"], observed=True)["gc_content"] \
        .agg(["count", "mean", "std", "min", "median", "max"]) \
        .reset_index().to_csv(OUT_GC_CSV, index=False)
    plot_gc_box(gc_df, OUT_GC_PLOT)
    print(f"GC图已保存: {OUT_GC_PLOT}")

    # ================== 3. k-mer PCC ==================
    kmer_df = compute_kmer_pcc(real_df, gen_df)
    kmer_df.to_csv(OUT_KMER_PCC_CSV, index=False)
    plot_kmer_pcc_bar(kmer_df, OUT_KMER_PLOT)
    print(f"k-mer图已保存: {OUT_KMER_PLOT}")

    # ================== 4. 散点图 ==================
    plot_oracle_scatter(real_df, OUT_DIR)
    plot_generated_oracle_scatter(real_df, gen_df, OUT_DIR)

    # ================== 5. 5-mer 全局散点图 ==================
    plot_kmer5_global_scatter(real_df, gen_df, OUT_DIR)


if __name__ == "__main__":
    main()