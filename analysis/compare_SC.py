# -*- coding: utf-8 -*-

import os
import sys
import gc
import re
import warnings
import logging
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 0. 静音与绘图风格
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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
    "axes.unicode_minus": False,
})

np.random.seed(42)


# ============================================================
# 1. 路径设置
# ============================================================

GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/high.txt"

PROMODGDE_LOW_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/SC_80bp/low_final_sequences_SC.txt"
PROMODGDE_MID_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/SC_80bp/medium_final_sequences_SC.txt"
PROMODGDE_HIGH_TXT = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/SC_80bp/high_final_sequences_SC.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"

ORACLE_DIR = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_MODEL_CONDITIONS = "defined_media"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/compare_SC1"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

LABEL_ORDER = ["low", "mid", "high"]
METHOD_ORDER = ["Real", "our", "PromoDGDE"]
GEN_METHODS = ["our", "PromoDGDE"]

# 公平比较：每个 method × label 下采样到相同数量
N_PER_METHOD_BIN = 1000

# 三分位阈值：运行时从真实数据 strength 自动计算
Q1 = None
Q2 = None

TARGET_RANGES = None

SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence", "sequence_std"]


# ============================================================
# 3. 基础函数
# ============================================================

def standardize_seq(s):
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ACGT"])


def extract_sequence_from_line(line):
    line = str(line).strip()
    if not line or line.startswith(">"):
        return ""

    tokens = re.split(r"[\s,\t;,\|]+", line)
    candidates = []

    for tok in tokens:
        tok_std = standardize_seq(tok)
        if len(tok_std) >= 20:
            candidates.append(tok_std)

    if candidates:
        return max(candidates, key=len)

    return standardize_seq(line)


def read_txt_sequences(path):
    if not os.path.exists(path):
        raise FileNotFoundError("未找到文件: {}".format(path))

    seqs = []
    with open(path, "r") as f:
        for line in f:
            seq = extract_sequence_from_line(line)
            if seq:
                seqs.append(seq)

    if len(seqs) == 0:
        raise ValueError("文件中没有有效序列: {}".format(path))

    return seqs


def find_seq_col(df):
    lower_map = {c.lower(): c for c in df.columns}
    for c in SEQ_COL_CANDIDATES:
        if c in lower_map:
            return lower_map[c]
    raise ValueError("找不到序列列，候选列为: {}".format(SEQ_COL_CANDIDATES))


def compute_tertile_thresholds_from_real_csv(path):
    """
    从真实数据 strength 中计算三分位阈值：
    low  : strength <= Q1
    mid  : Q1 < strength <= Q2
    high : strength > Q2
    """
    if not os.path.exists(path):
        raise FileNotFoundError("未找到真实数据文件: {}".format(path))

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "strength" not in df.columns:
        raise ValueError("真实数据中缺少 strength 列")

    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    df = df.dropna(subset=["strength"]).copy()

    q1 = float(df["strength"].quantile(1.0 / 3.0))
    q2 = float(df["strength"].quantile(2.0 / 3.0))

    return q1, q2


def assign_label_by_strength(x):
    global Q1, Q2

    if Q1 is None or Q2 is None:
        raise ValueError("Q1/Q2 还没有初始化，请先在 main() 中计算三分位阈值。")

    if x <= Q1:
        return "low"
    elif x <= Q2:
        return "mid"
    else:
        return "high"


def target_success(score, label):
    global TARGET_RANGES

    if TARGET_RANGES is None:
        raise ValueError("TARGET_RANGES 还没有初始化。")

    lo, hi = TARGET_RANGES[label]
    return (score > lo) and (score <= hi)


def gc_content(seq):
    seq = standardize_seq(seq)
    if len(seq) == 0:
        return np.nan
    return float(seq.count("G") + seq.count("C")) / float(len(seq))


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


def downsample_per_method_label(df, n_per_group=1000, seed=42):
    parts = []
    for (method, label), sub in df.groupby(["method", "label"]):
        if n_per_group is not None and len(sub) > n_per_group:
            sub = sub.sample(n=n_per_group, random_state=seed)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ============================================================
# 4. 数据读取
# ============================================================

def load_real_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("未找到真实数据文件: {}".format(path))

    df = pd.read_csv(path)
    original_cols = df.columns.tolist()
    df.columns = [c.strip().lower() for c in df.columns]

    if "strength" not in df.columns:
        raise ValueError("真实数据中缺少 strength 列，当前列名: {}".format(original_cols))

    seq_col = find_seq_col(df)

    df["sequence"] = df[seq_col].astype(str).apply(standardize_seq)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
    else:
        df["label"] = df["strength"].apply(assign_label_by_strength)

    df = df.dropna(subset=["sequence", "strength", "label"]).copy()
    df = df[df["sequence"].str.len() > 0].copy()

    # 用你指定的三档重新筛
    df["label"] = df["strength"].apply(assign_label_by_strength)
    df = df[df["label"].isin(LABEL_ORDER)].copy()

    df["method"] = "Real"
    df["real_exp_strength"] = df["strength"]

    return df[["method", "label", "sequence", "real_exp_strength"]].reset_index(drop=True)


def load_generated_data():
    rows = []

    path_map = {
        ("our", "low"): GEN_LOW_TXT,
        ("our", "mid"): GEN_MID_TXT,
        ("our", "high"): GEN_HIGH_TXT,
        ("PromoDGDE", "low"): PROMODGDE_LOW_TXT,
        ("PromoDGDE", "mid"): PROMODGDE_MID_TXT,
        ("PromoDGDE", "high"): PROMODGDE_HIGH_TXT,
    }

    for (method, label), path in path_map.items():
        seqs = read_txt_sequences(path)
        for seq in seqs:
            rows.append({
                "method": method,
                "label": label,
                "sequence": seq,
            })

    return pd.DataFrame(rows)


# ============================================================
# 5. Oracle 预测
# ============================================================

def load_oracle_once(oracle_dir, model_conditions):
    import tensorflow as tf

    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)

    from aux import load_model

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    graph = tf.Graph()
    with graph.as_default():
        model, scaler, batch_size = load_model(model_conditions)

    return graph, model, scaler, batch_size


def oracle_predict_with_loaded_model(sequences, graph, model, scaler, batch_size):
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)

    from aux import evaluate_model

    preds = evaluate_model(sequences, model, scaler, batch_size, graph)
    return np.asarray(preds, dtype=float).reshape(-1)


def attach_oracle_scores(df, oracle_dir, model_conditions, score_col="oracle_score"):
    print("Loading oracle predictor ...")
    graph, model, scaler, batch_size = load_oracle_once(oracle_dir, model_conditions)

    parts = []
    for (method, label), sub in df.groupby(["method", "label"]):
        sub = sub.copy()
        seqs = sub["sequence"].tolist()
        print("Scoring: {} | {} | n={}".format(method, label, len(seqs)))

        preds = oracle_predict_with_loaded_model(
            sequences=seqs,
            graph=graph,
            model=model,
            scaler=scaler,
            batch_size=batch_size,
        )

        sub[score_col] = preds
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


# ============================================================
# 6. 指标计算
# ============================================================

def compute_target_success(df):
    rows = []

    for method in GEN_METHODS:
        for label in LABEL_ORDER:
            sub = df[(df["method"] == method) & (df["label"] == label)].copy()
            scores = pd.to_numeric(sub["oracle_score"], errors="coerce").dropna().values
            if len(scores) == 0:
                continue

            success = [target_success(s, label) for s in scores]

            rows.append({
                "method": method,
                "label": label,
                "n": len(scores),
                "target_success_rate": float(np.mean(success)),
                "oracle_mean": float(np.mean(scores)),
                "oracle_std": float(np.std(scores)),
                "oracle_median": float(np.median(scores)),
            })

    return pd.DataFrame(rows)


def compute_monotonicity(df):
    rows = []

    for method in GEN_METHODS:
        means = {}
        medians = {}

        for label in LABEL_ORDER:
            sub = df[(df["method"] == method) & (df["label"] == label)]
            vals = pd.to_numeric(sub["oracle_score"], errors="coerce").dropna().values

            means[label] = float(np.mean(vals)) if len(vals) else np.nan
            medians[label] = float(np.median(vals)) if len(vals) else np.nan

        rows.append({
            "method": method,
            "low_mean": means["low"],
            "mid_mean": means["mid"],
            "high_mean": means["high"],
            "mean_monotonic_low_mid_high": bool(means["low"] < means["mid"] < means["high"]),
            "low_median": medians["low"],
            "mid_median": medians["mid"],
            "high_median": medians["high"],
            "median_monotonic_low_mid_high": bool(medians["low"] < medians["mid"] < medians["high"]),
        })

    return pd.DataFrame(rows)


def compute_gc_summary(df):
    df = df.copy()
    df["gc_content"] = df["sequence"].apply(gc_content)

    rows = []
    for method in METHOD_ORDER:
        for label in LABEL_ORDER:
            sub = df[(df["method"] == method) & (df["label"] == label)]
            vals = pd.to_numeric(sub["gc_content"], errors="coerce").dropna()
            if len(vals) == 0:
                continue

            rows.append({
                "method": method,
                "label": label,
                "n": len(vals),
                "gc_mean": vals.mean(),
                "gc_std": vals.std(),
                "gc_median": vals.median(),
                "gc_min": vals.min(),
                "gc_max": vals.max(),
            })

    return pd.DataFrame(rows)


def compute_gc_fidelity(gc_summary):
    rows = []
    for method in GEN_METHODS:
        for label in LABEL_ORDER:
            real_row = gc_summary[(gc_summary["method"] == "Real") & (gc_summary["label"] == label)]
            gen_row = gc_summary[(gc_summary["method"] == method) & (gc_summary["label"] == label)]

            if len(real_row) == 0 or len(gen_row) == 0:
                continue

            real_gc = float(real_row["gc_mean"].iloc[0])
            gen_gc = float(gen_row["gc_mean"].iloc[0])

            rows.append({
                "method": method,
                "label": label,
                "real_gc_mean": real_gc,
                "gen_gc_mean": gen_gc,
                "gc_abs_diff": abs(gen_gc - real_gc),
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. k-mer 计算（只到 5）
# ============================================================

def all_kmers(k):
    return ["".join(p) for p in product("ACGT", repeat=k)]


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
            if set(kmer).issubset(set("ACGT")):
                counter[kmer] += 1
                total += 1

    if total == 0:
        return np.zeros(len(vocab), dtype=float)

    return np.array([counter[km] / float(total) for km in vocab], dtype=float)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_kmer_fidelity(df, ks=(3, 4, 5)):
    rows = []

    for label in LABEL_ORDER:
        real_seqs = df[(df["method"] == "Real") & (df["label"] == label)]["sequence"].tolist()

        for method in GEN_METHODS:
            gen_seqs = df[(df["method"] == method) & (df["label"] == label)]["sequence"].tolist()

            for k in ks:
                real_vec = kmer_freq_vector(real_seqs, k)
                gen_vec = kmer_freq_vector(gen_seqs, k)

                rows.append({
                    "method": method,
                    "label": label,
                    "k": k,
                    "kmer_corr": safe_pearson(real_vec, gen_vec),
                    "n_real": len(real_seqs),
                    "n_gen": len(gen_seqs),
                })

    return pd.DataFrame(rows)


# ============================================================
# 8. Pareto 数据（不做综合得分）
# ============================================================

def build_pareto_table(success_df, kmer_df):
    kmer_avg = (
        kmer_df.groupby(["method", "label"])["kmer_corr"]
        .mean()
        .reset_index()
        .rename(columns={"kmer_corr": "mean_kmer_corr_345"})
    )

    pareto_df = success_df.merge(kmer_avg, on=["method", "label"], how="left")
    return pareto_df


# ============================================================
# 9. 画图函数
# ============================================================

def plot_score_distribution(df, out_path):
    plot_rows = []

    real = df[df["method"] == "Real"].copy()
    for _, row in real.iterrows():
        plot_rows.append({
            "label": row["label"],
            "source": "Real-Exp",
            "value": row["real_exp_strength"],
        })
        plot_rows.append({
            "label": row["label"],
            "source": "Real-Oracle",
            "value": row["oracle_score"],
        })

    for method, source_name in [
        ("our", "our-Oracle"),
        ("PromoDGDE", "PromoDGDE-Oracle"),
    ]:
        sub = df[df["method"] == method]
        for _, row in sub.iterrows():
            plot_rows.append({
                "label": row["label"],
                "source": source_name,
                "value": row["oracle_score"],
            })

    plot_df = pd.DataFrame(plot_rows)
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=LABEL_ORDER, ordered=True)

    source_order = ["Real-Exp", "Real-Oracle", "our-Oracle", "PromoDGDE-Oracle"]
    color_map = {
        "Real-Exp": "#4D4D4D",
        "Real-Oracle": "#0072B2",
        "our-Oracle": "#D55E00",
        "PromoDGDE-Oracle": "#009E73",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    base_pos = {"low": 1.0, "mid": 2.0, "high": 3.0}
    offsets = {
        "Real-Exp": -0.27,
        "Real-Oracle": -0.09,
        "our-Oracle": 0.09,
        "PromoDGDE-Oracle": 0.27,
    }

    for label in LABEL_ORDER:
        for source in source_order:
            vals = plot_df[(plot_df["label"] == label) & (plot_df["source"] == source)]["value"].dropna().values
            if len(vals) == 0:
                continue

            pos = base_pos[label] + offsets[source]

            ax.boxplot(
                vals,
                positions=[pos],
                widths=0.14,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color_map[source], alpha=0.28, edgecolor=color_map[source], linewidth=1.1),
                medianprops=dict(color=color_map[source], linewidth=1.3),
                whiskerprops=dict(color=color_map[source], linewidth=0.8),
                capprops=dict(color=color_map[source], linewidth=0.8),
            )

    handles = [Patch(facecolor=color_map[s], edgecolor=color_map[s], alpha=0.5, label=s) for s in source_order]
    ax.legend(handles=handles, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.1))

    ax.set_xlim(0.55, 3.45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_xlabel("Target condition")
    ax.set_ylabel("Expression strength / Oracle score")
    ax.set_title("Expression controllability comparison")

    lims = get_full_limits(plot_df["value"], pad_ratio=0.06, min_pad=0.5)
    if lims:
        ax.set_ylim(*lims)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def plot_success_rate(success_df, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    x = np.arange(len(LABEL_ORDER))
    width = 0.34

    color_map = {
        "our": "#D55E00",
        "PromoDGDE": "#009E73",
    }

    for i, method in enumerate(GEN_METHODS):
        vals = []
        for label in LABEL_ORDER:
            sub = success_df[(success_df["method"] == method) & (success_df["label"] == label)]
            vals.append(float(sub["target_success_rate"].iloc[0]) if len(sub) else np.nan)

        xs = x + (i - 0.5) * width
        bars = ax.bar(
            xs, vals, width=width,
            color=color_map[method], alpha=0.82,
            edgecolor="black", linewidth=0.5,
            label=method.replace("_", " "),
        )

        for bar, val in zip(bars, vals):
            if not pd.isna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + 0.025,
                    "{:.2f}".format(val),
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Target condition")
    ax.set_ylabel("Target success rate")
    ax.set_title("Conditional target success")
    ax.legend(loc="upper right", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def plot_gc_distribution(df, out_path):
    df = df.copy()
    df["gc_content"] = df["sequence"].apply(gc_content)

    color_map = {
        "Real": "#0072B2",
        "our": "#D55E00",
        "PromoDGDE": "#009E73",
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    base_pos = {"low": 1.0, "mid": 2.0, "high": 3.0}
    offsets = {
        "Real": -0.22,
        "our": 0.0,
        "PromoDGDE": 0.22,
    }

    for label in LABEL_ORDER:
        for method in METHOD_ORDER:
            vals = df[(df["label"] == label) & (df["method"] == method)]["gc_content"].dropna().values
            if len(vals) == 0:
                continue

            pos = base_pos[label] + offsets[method]

            ax.boxplot(
                vals,
                positions=[pos],
                widths=0.16,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color_map[method], alpha=0.28, edgecolor=color_map[method], linewidth=1.1),
                medianprops=dict(color=color_map[method], linewidth=1.3),
                whiskerprops=dict(color=color_map[method], linewidth=0.8),
                capprops=dict(color=color_map[method], linewidth=0.8),
            )

    handles = [
        Patch(facecolor=color_map[m], edgecolor=color_map[m], alpha=0.5, label=m.replace("_", " "))
        for m in METHOD_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    ax.set_xlim(0.55, 3.45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_xlabel("Target condition")
    ax.set_ylabel("GC content")
    ax.set_title("GC distribution fidelity")

    lims = get_full_limits(df["gc_content"], pad_ratio=0.08, min_pad=0.02)
    if lims:
        ax.set_ylim(*lims)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def plot_kmer_bar(kmer_df, out_path):
    color_map = {
        "our": "#D55E00",
        "PromoDGDE": "#009E73",
    }

    ks = [3, 4, 5]
    fig, axes = plt.subplots(1, len(ks), figsize=(9.2, 3.6), sharey=True)

    for ax, k in zip(axes, ks):
        sub_k = kmer_df[kmer_df["k"] == k]

        x = np.arange(len(LABEL_ORDER))
        width = 0.34

        for i, method in enumerate(GEN_METHODS):
            vals = []
            for label in LABEL_ORDER:
                sub = sub_k[(sub_k["method"] == method) & (sub_k["label"] == label)]
                vals.append(float(sub["kmer_corr"].iloc[0]) if len(sub) else np.nan)

            xs = x + (i - 0.5) * width
            bars = ax.bar(
                xs, vals, width=width,
                color=color_map[method], alpha=0.82,
                edgecolor="black", linewidth=0.5,
                label=method.replace("_", " "),
            )

            for bar, val in zip(bars, vals):
                if not pd.isna(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        val + 0.01,
                        "{:.3f}".format(val),
                        ha="center", va="bottom", fontsize=6
                    )

        ax.set_title("{}-mer".format(k))
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_ORDER)
        ax.set_xlabel("Condition")
        ax.set_ylim(0, 1.08)

        if k == 3:
            ax.set_ylabel("Pearson correlation with real bin")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.25))
    fig.suptitle("k-mer distributional fidelity", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def plot_pareto(pareto_df, out_path):
    color_map = {
        "our": "#D55E00",
        "PromoDGDE": "#009E73",
    }

    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    for method in GEN_METHODS:
        sub = pareto_df[pareto_df["method"] == method]

        ax.scatter(
            sub["target_success_rate"],
            sub["mean_kmer_corr_345"],
            s=90,
            color=color_map[method],
            alpha=0.78,
            edgecolors="black",
            linewidths=0.5,
            label=method.replace("_", " "),
        )

        for _, row in sub.iterrows():
            ax.text(
                row["target_success_rate"] + 0.012,
                row["mean_kmer_corr_345"] + 0.006,
                row["label"],
                fontsize=8,
            )

    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Target success rate")
    ax.set_ylabel("Mean 3/4/5-mer correlation")
    ax.set_title("Controllability-fidelity comparison")
    ax.legend(loc="upper left", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# 10. 解释文件（不输出综合胜者）
# ============================================================

def write_interpretation(success_df, monotonic_df, gc_fid_df, pareto_df, out_txt):
    with open(out_txt, "w") as f:
        f.write("End-to-end vs generate-then-optimize comparison\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Target success rate\n")
        f.write(success_df.to_string(index=False))
        f.write("\n\n")

        f.write("2. Monotonicity low < mid < high\n")
        f.write(monotonic_df.to_string(index=False))
        f.write("\n\n")

        f.write("3. GC fidelity (smaller gc_abs_diff is better)\n")
        f.write(gc_fid_df.to_string(index=False))
        f.write("\n\n")

        f.write("4. Pareto table (higher success rate and higher mean_kmer_corr_345 are better)\n")
        f.write(pareto_df.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide:\n")
        f.write("- If PromoDGDE has higher target_success_rate, it means post-hoc optimization is stronger at pushing sequences into the target score interval.\n")
        f.write("- If our has higher k-mer fidelity and smaller GC deviation, it means end-to-end generation better preserves the natural promoter distribution.\n")
        f.write("- If our also shows monotonic low < mid < high, then it supports the claim that the model directly learned conditional generation rather than relying on post-hoc optimization.\n")
        f.write("- Therefore, do not claim superiority using a single metric. Instead, compare controllability and fidelity jointly.\n")


# ============================================================
# 11. 主函数
# ============================================================

def main():
    global Q1, Q2, TARGET_RANGES

    Q1, Q2 = compute_tertile_thresholds_from_real_csv(REAL_CSV)

    TARGET_RANGES = {
        "low":  (-np.inf, Q1),
        "mid":  (Q1, Q2),
        "high": (Q2, np.inf),
    }

    print("\n==============================")
    print("Tertile thresholds")
    print("==============================")
    print("Q1 / low-mid threshold  = {:.6f}".format(Q1))
    print("Q2 / mid-high threshold = {:.6f}".format(Q2))
    print("low  : strength <= {:.6f}".format(Q1))
    print("mid  : {:.6f} < strength <= {:.6f}".format(Q1, Q2))
    print("high : strength > {:.6f}".format(Q2))

    print("\n==============================")
    print("Loading data")
    print("==============================")

    real_df = load_real_data(REAL_CSV)
    gen_df = load_generated_data()

    print("\nRaw data counts:")
    print(pd.concat([real_df[["method", "label", "sequence"]], gen_df], ignore_index=True)
          .groupby(["method", "label"])
          .size())

    all_df = pd.concat([
        real_df[["method", "label", "sequence", "real_exp_strength"]],
        gen_df.assign(real_exp_strength=np.nan)[["method", "label", "sequence", "real_exp_strength"]],
    ], ignore_index=True)

    all_df = downsample_per_method_label(
        all_df,
        n_per_group=N_PER_METHOD_BIN,
        seed=42,
    )

    print("\nAfter fair downsampling:")
    print(all_df.groupby(["method", "label"]).size())

    all_df.to_csv(os.path.join(OUT_DIR, "all_sequences_after_downsampling_before_oracle.csv"), index=False)

    print("\n==============================")
    print("Oracle scoring")
    print("==============================")

    all_df = attach_oracle_scores(
        all_df,
        oracle_dir=ORACLE_DIR,
        model_conditions=ORACLE_MODEL_CONDITIONS,
        score_col="oracle_score",
    )

    all_df["gc_content"] = all_df["sequence"].apply(gc_content)
    all_df.to_csv(os.path.join(OUT_DIR, "all_sequences_with_oracle_scores.csv"), index=False)

    print("\n==============================")
    print("Computing metrics")
    print("==============================")

    success_df = compute_target_success(all_df)
    success_df.to_csv(os.path.join(OUT_DIR, "target_success_rate.csv"), index=False)

    monotonic_df = compute_monotonicity(all_df)
    monotonic_df.to_csv(os.path.join(OUT_DIR, "monotonicity_low_mid_high.csv"), index=False)

    gc_summary = compute_gc_summary(all_df)
    gc_summary.to_csv(os.path.join(OUT_DIR, "gc_summary.csv"), index=False)

    gc_fid_df = compute_gc_fidelity(gc_summary)
    gc_fid_df.to_csv(os.path.join(OUT_DIR, "gc_fidelity.csv"), index=False)

    kmer_df = compute_kmer_fidelity(all_df, ks=(3, 4, 5))
    kmer_df.to_csv(os.path.join(OUT_DIR, "kmer_fidelity_345.csv"), index=False)

    pareto_df = build_pareto_table(success_df, kmer_df)
    pareto_df.to_csv(os.path.join(OUT_DIR, "pareto_table.csv"), index=False)

    write_interpretation(
        success_df=success_df,
        monotonic_df=monotonic_df,
        gc_fid_df=gc_fid_df,
        pareto_df=pareto_df,
        out_txt=os.path.join(OUT_DIR, "method_comparison_interpretation.txt"),
    )

    print("\n==============================")
    print("Plotting figures")
    print("==============================")

    plot_score_distribution(
        all_df,
        os.path.join(OUT_DIR, "Fig1_expression_score_distribution.png"),
    )

    plot_success_rate(
        success_df,
        os.path.join(OUT_DIR, "Fig2_target_success_rate.png"),
    )

    plot_gc_distribution(
        all_df,
        os.path.join(OUT_DIR, "Fig3_gc_distribution.png"),
    )

    plot_kmer_bar(
        kmer_df,
        os.path.join(OUT_DIR, "Fig4_kmer_fidelity_bar.png"),
    )

    plot_pareto(
        pareto_df,
        os.path.join(OUT_DIR, "Fig5_controllability_fidelity_pareto.png"),
    )

    print("\n==============================")
    print("Finished")
    print("==============================")
    print("Results saved to:")
    print(OUT_DIR)

    print("\nTarget success:")
    print(success_df)

    print("\nMonotonicity:")
    print(monotonic_df)

    print("\nGC fidelity:")
    print(gc_fid_df)

    print("\nPareto table:")
    print(pareto_df)


if __name__ == "__main__":
    main()