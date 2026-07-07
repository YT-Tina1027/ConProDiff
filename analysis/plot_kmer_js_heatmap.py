import os
import warnings
import logging
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# ================== 静音设置 ==================
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ================== 全局绘图风格 ==================
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False
})

np.random.seed(42)

# ================== 路径配置 ==================
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/high.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/defined_SC_Ura_core80_812.csv"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/analysis_kmer_js1"
os.makedirs(OUT_DIR, exist_ok=True)

KMER_K = 6
EPS = 1e-8

OUT_JS_MATRIX_CSV = os.path.join(OUT_DIR, f"kmer_{KMER_K}mer_js_matrix.csv")
OUT_JS_HEATMAP_PNG = os.path.join(OUT_DIR, f"kmer_{KMER_K}mer_js_heatmap.png")
OUT_JS_HEATMAP_PDF = os.path.join(OUT_DIR, f"kmer_{KMER_K}mer_js_heatmap.pdf")

SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]
LABEL_ORDER = ["low", "mid", "high"]


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


def load_generated_txts(low_txt, mid_txt, high_txt):
    rows = []
    for seq in read_txt_sequences(low_txt):
        rows.append({"group": "opt_low", "sequence_std": seq})
    for seq in read_txt_sequences(mid_txt):
        rows.append({"group": "opt_mid", "sequence_std": seq})
    for seq in read_txt_sequences(high_txt):
        rows.append({"group": "opt_high", "sequence_std": seq})
    return pd.DataFrame(rows)


def load_real_strength_df_from_label(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到真实数据文件: {path}")

    real_df = pd.read_csv(path)
    real_df.columns = [c.strip().lower() for c in real_df.columns]

    if "label" not in real_df.columns:
        raise ValueError("真实数据中缺少 label 列")

    seq_col = find_seq_col(real_df)

    real_df = real_df.copy()
    real_df["label"] = real_df["label"].astype(str).str.strip().str.lower()
    real_df["sequence_std"] = real_df[seq_col].astype(str).apply(standardize_seq)

    real_df = real_df.dropna(subset=["label", "sequence_std"]).reset_index(drop=True)
    real_df = real_df[real_df["sequence_std"].str.len() > 0]
    real_df = real_df[real_df["label"].isin(LABEL_ORDER)]
    real_df["group"] = real_df["label"].map(lambda x: f"real_{x}")

    return real_df.reset_index(drop=True)


# ================== k-mer 分布 ==================
def all_kmers(k):
    return ["".join(p) for p in product("ATCG", repeat=k)]


def kmer_freq_vector(seqs, k, eps=1e-8):
    vocab = all_kmers(k)
    counter = Counter()
    total = 0

    for seq in seqs:
        seq = standardize_seq(seq)
        if len(seq) < k:
            continue
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if set(kmer).issubset("ATCG"):
                counter[kmer] += 1
                total += 1

    vec = np.array([counter[km] for km in vocab], dtype=float)
    vec = vec + eps
    vec = vec / vec.sum()
    return vec


def js_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2.0))


# ================== JS 计算 ==================
def build_group_distributions(real_df, gen_df, k):
    group_dists = {}

    for label in LABEL_ORDER:
        real_seqs = real_df.loc[real_df["group"] == f"real_{label}", "sequence_std"].tolist()
        gen_seqs = gen_df.loc[gen_df["group"] == f"opt_{label}", "sequence_std"].tolist()

        if len(real_seqs) == 0:
            raise ValueError(f"real_{label} 没有有效序列")
        if len(gen_seqs) == 0:
            raise ValueError(f"opt_{label} 没有有效序列")

        group_dists[f"real_{label}"] = kmer_freq_vector(real_seqs, k=k, eps=EPS)
        group_dists[f"gen_{label}"] = kmer_freq_vector(gen_seqs, k=k, eps=EPS)

    return group_dists


def build_js_matrix(group_dists):
    js_mat = pd.DataFrame(index=LABEL_ORDER, columns=LABEL_ORDER, dtype=float)

    for gen_label in LABEL_ORDER:
        for real_label in LABEL_ORDER:
            d_gen = group_dists[f"gen_{gen_label}"]
            d_real = group_dists[f"real_{real_label}"]
            js_mat.loc[gen_label, real_label] = js_distance(d_gen, d_real)

    return js_mat


# ================== 绘图 ==================
def plot_js_heatmap(js_df, out_png, out_pdf, k):
    mat = js_df.loc[LABEL_ORDER, LABEL_ORDER].values

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(mat, cmap="Blues", aspect="equal")

    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels([f"real_{x}" for x in LABEL_ORDER])
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_yticklabels([f"gen_{x}" for x in LABEL_ORDER])

    ax.set_xlabel("Real sequence group")
    ax.set_ylabel("Generated sequence group")
    ax.set_title(f"JS distance of {k}-mer distributions", pad=8)

    threshold = (mat.max() + mat.min()) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("JS distance")

    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ================== 主函数 ==================
def main():
    gen_df = load_generated_txts(GEN_LOW_TXT, GEN_MID_TXT, GEN_HIGH_TXT)
    real_df = load_real_strength_df_from_label(REAL_CSV)

    group_dists = build_group_distributions(real_df, gen_df, k=KMER_K)
    js_mat = build_js_matrix(group_dists)

    js_mat.to_csv(OUT_JS_MATRIX_CSV)
    plot_js_heatmap(js_mat, OUT_JS_HEATMAP_PNG, OUT_JS_HEATMAP_PDF, k=KMER_K)

    print(f"Saved JS matrix: {OUT_JS_MATRIX_CSV}")
    print(f"Saved heatmap: {OUT_JS_HEATMAP_PNG}")
    print(f"Saved heatmap: {OUT_JS_HEATMAP_PDF}")


if __name__ == "__main__":
    main()