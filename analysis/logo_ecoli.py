import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

# =========================
# 静音设置
# =========================
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# =========================
# 全局绘图风格（Nature-like）
# =========================
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False
})

np.random.seed(42)

# =========================
# 路径配置（E. coli）
# =========================
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_ecoli1_CFG_gc3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_ecoli1_CFG_gc3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_ecoli1_CFG_gc3.0/high.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/E_coli_promoter_processed/ecoli_exp_labeled_split.csv"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/analysis_logo_compare_ecoli_gc"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_LOGO_PNG = os.path.join(OUT_DIR, "motif_logo_real_vs_generated_ecoli_bits.png")
OUT_LOGO_PDF = os.path.join(OUT_DIR, "motif_logo_real_vs_generated_ecoli_bits.pdf")

LABEL_ORDER = ["low", "mid", "high"]
SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]

# 为了公平和画图速度，真实/生成每组最多抽这么多条
MAX_SEQS_PER_GROUP = 5000

# True: 信息量 logo（Bits）
# False: 概率 logo（Frequency）
USE_INFORMATION_LOGO = True

# 如果你只想看 test / val / train，可以设成 "test"、"val"、"train"
# 不筛选就设成 None
REAL_SPLIT_FILTER = None


# =========================
# 辅助函数
# =========================
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
            if (not line) or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if seq:
                seqs.append(seq)

    if not seqs:
        raise ValueError(f"文件中无有效序列: {path}")
    return seqs


def sample_sequences(seqs, max_n=5000, seed=42):
    seqs = list(seqs)
    if len(seqs) <= max_n:
        return seqs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=max_n, replace=False)
    return [seqs[i] for i in idx]


def filter_same_length(seqs):
    seqs = [standardize_seq(s) for s in seqs]
    seqs = [s for s in seqs if len(s) > 0]
    if not seqs:
        return []

    lengths = pd.Series([len(s) for s in seqs])
    main_len = int(lengths.mode().iloc[0])
    seqs = [s for s in seqs if len(s) == main_len]
    return seqs


def sequences_to_probability_df(seqs):
    if len(seqs) == 0:
        raise ValueError("空序列列表，无法构建 logo")

    seqs = filter_same_length(seqs)
    if len(seqs) == 0:
        raise ValueError("过滤后没有等长有效序列")

    L = len(seqs[0])
    mat = np.zeros((L, 4), dtype=float)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    for seq in seqs:
        for i, ch in enumerate(seq):
            if ch in base_to_idx:
                mat[i, base_to_idx[ch]] += 1.0

    mat = mat / mat.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, columns=["A", "C", "G", "T"])


def probability_to_information_df(prob_df: pd.DataFrame):
    p = prob_df[["A", "C", "G", "T"]].values
    eps = 1e-12
    entropy = -np.sum(p * np.log2(p + eps), axis=1)
    info = 2.0 - entropy
    info = np.clip(info, 0, None)
    logo_mat = p * info[:, None]
    return pd.DataFrame(logo_mat, columns=["A", "C", "G", "T"])


def build_logo_matrix(seqs, use_information_logo=True):
    prob_df = sequences_to_probability_df(seqs)
    if use_information_logo:
        return probability_to_information_df(prob_df)
    return prob_df


# =========================
# 数据加载
# =========================
def load_generated_groups():
    return {
        "low": read_txt_sequences(GEN_LOW_TXT),
        "mid": read_txt_sequences(GEN_MID_TXT),
        "high": read_txt_sequences(GEN_HIGH_TXT),
    }


def load_real_groups():
    if not os.path.exists(REAL_CSV):
        raise FileNotFoundError(f"未找到真实数据文件: {REAL_CSV}")

    df = pd.read_csv(REAL_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("真实 CSV 中缺少 label 列")

    seq_col = find_seq_col(df)

    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["sequence_std"] = df[seq_col].astype(str).apply(standardize_seq)

    if REAL_SPLIT_FILTER is not None:
        if "split" not in df.columns:
            raise ValueError("设置了 REAL_SPLIT_FILTER，但真实 CSV 中没有 split 列")
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        df = df[df["split"] == REAL_SPLIT_FILTER].copy()

    df = df[df["label"].isin(LABEL_ORDER)]
    df = df[df["sequence_std"].str.len() > 0].reset_index(drop=True)

    groups = {}
    for label in LABEL_ORDER:
        groups[label] = df.loc[df["label"] == label, "sequence_std"].tolist()

    return groups


# =========================
# 画图
# =========================
def draw_logo(ax, logo_df, title=None, row_label=None, ymax=None, show_xlabel=True):
    color_scheme = {
        "A": "#2E8B57",
        "C": "#1F77B4",
        "G": "#F39C12",
        "T": "#D62728",
    }

    logomaker.Logo(
        logo_df,
        ax=ax,
        color_scheme=color_scheme,
        fade_below=0.5,
        shade_below=0.0,
        width=0.9,
        center_values=False
    )

    if title is not None:
        ax.set_title(title, pad=8)

    if ymax is not None:
        ax.set_ylim(0, ymax)

    if show_xlabel:
        ax.set_xlabel("Position", labelpad=5)
    else:
        ax.set_xlabel("")

    ax.set_ylabel(row_label if row_label else "", labelpad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===== 关键修改：动态控制刻度，避免最后两个重叠 =====
    L = logo_df.shape[0]

    if L <= 100:
        step = 10
    elif L <= 180:
        step = 20
    else:
        step = 25

    xticks = list(range(0, L, step))

    # 只在“离最后一个主刻度足够远”时，才额外加末位
    if (L - 1) not in xticks:
        if len(xticks) == 0 or (L - 1 - xticks[-1]) >= step * 0.6:
            xticks.append(L - 1)
        else:
            # 如果太近，就直接把最后一个主刻度替换成末位
            xticks[-1] = L - 1

    xticks = sorted(set(xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x + 1) for x in xticks])
    ax.tick_params(axis="x", pad=3)

    ax.set_box_aspect(1 / 6.5)


def main():
    real_groups = load_real_groups()
    gen_groups = load_generated_groups()

    real_groups = {
        k: filter_same_length(sample_sequences(v, MAX_SEQS_PER_GROUP, seed=42))
        for k, v in real_groups.items()
    }
    gen_groups = {
        k: filter_same_length(sample_sequences(v, MAX_SEQS_PER_GROUP, seed=42))
        for k, v in gen_groups.items()
    }

    for k in LABEL_ORDER:
        print(f"Real {k}: {len(real_groups[k])} sequences")
        print(f"Generated {k}: {len(gen_groups[k])} sequences")

    logo_mats = {}
    for source_name, groups in [("Real", real_groups), ("Generated", gen_groups)]:
        for label in LABEL_ORDER:
            seqs = groups[label]
            if len(seqs) == 0:
                raise ValueError(f"{source_name}-{label} 组没有有效序列")
            logo_mats[(source_name, label)] = build_logo_matrix(
                seqs,
                use_information_logo=USE_INFORMATION_LOGO
            )

    # 每一列单独统一 y 轴：同条件下 real vs generated 可比
    col_ymax = {}
    for label in LABEL_ORDER:
        vals = [
            float(logo_mats[("Real", label)].sum(axis=1).max()),
            float(logo_mats[("Generated", label)].sum(axis=1).max())
        ]
        ymax = max(vals)
        if ymax <= 0:
            ymax = 2.0 if USE_INFORMATION_LOGO else 1.0
        col_ymax[label] = ymax * 1.0

    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(12, 2.4),
        sharex=False,
        sharey=False
    )

    row_names = ["Real", "Generated"]

    for i, source_name in enumerate(row_names):
        for j, label in enumerate(LABEL_ORDER):
            ax = axes[i, j]
            draw_logo(
                ax=ax,
                logo_df=logo_mats[(source_name, label)],
                title=label if i == 0 else None,
                row_label=source_name if j == 0 else None,
                ymax=col_ymax[label],
                show_xlabel=(i == 1)
            )

    unit_label = "Bits" if USE_INFORMATION_LOGO else "Frequency"
    axes[0, 0].text(
        -0.12, 1.05, unit_label,
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.20,
        top=0.90,
        wspace=0.18,
        hspace=0.65
    )

    fig.savefig(OUT_LOGO_PNG, dpi=600, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT_LOGO_PDF, bbox_inches="tight", pad_inches=0.12)
    plt.close()

    print(f"Saved: {OUT_LOGO_PNG}")
    print(f"Saved: {OUT_LOGO_PDF}")


if __name__ == "__main__":
    main()