"""
GC-Strength Correlation Analysis — 4 separate figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ══════════════════════════════════════════════════════════════
# 0. 样式
# ══════════════════════════════════════════════════════════════
NATURE_RC = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial"],
    "font.size":             9,
    "axes.labelsize":        9,
    "axes.titlesize":        9,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       8,
    "legend.frameon":        False,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "axes.unicode_minus":    False,
    "font.weight":           "normal",
    "axes.labelweight":      "normal",
    "axes.titleweight":      "normal",
}

# 箱线图颜色（蓝/绿/红，原版样式）
BIN_COLORS = {"low": "#4C72B0", "mid": "#55A868", "high": "#C44E52"}
BIN_ORDER  = ["low", "mid", "high"]
FIG_WIDTH  = 4.0  # 宽度 (英寸)
FIG_HEIGHT = 3.5  # 高度 (英寸)


# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

def compute_gc(seq: str) -> float:
    seq = str(seq).upper()
    seq = "".join(c for c in seq if c in "ACGT")
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")


def load_and_prepare(path: str, num_bins: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["sequence", "strength"]].dropna()
    df["sequence"] = df["sequence"].str.strip().str.upper()
    df = df[~df["sequence"].str.contains("N")]
    df["gc"] = df["sequence"].apply(compute_gc)
    df = df.dropna(subset=["gc"])
    df["bin"] = pd.qcut(df["strength"], q=num_bins, labels=BIN_ORDER)
    return df


def analyze(df: pd.DataFrame, label: str):
    r, p = stats.pearsonr(df["gc"], df["strength"])
    print(f"\n[{label}]  N={len(df)}  r={r:.4f}  p={p:.2e}")
    for b in BIN_ORDER:
        sub = df[df["bin"] == b]["gc"]
        print(f"  {b:4s}: mean GC={sub.mean():.4f} ± {sub.std():.4f}  n={len(sub)}")
    f, p_a = stats.f_oneway(*[df[df["bin"] == b]["gc"].values for b in BIN_ORDER])
    print(f"  ANOVA: F={f:.2f}  p={p_a:.2e}")
    return r, p


def save_fig(fig, path_stem):
    for ext in ("pdf", "png"):
        p = f"{path_stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  [Saved] {p}")
    plt.close(fig)


# ─────────────────────────────────────────
# Fig A/C — 箱线图（原版样式）
# ─────────────────────────────────────────

def plot_boxplot(df, name, r, path_stem):
    """
    原版箱线图样式：
    - 彩色填充箱体，黑色边框/须线/帽线
    - 显示异常值（小圆点）
    - 菱形标注均值
    """
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

        data_by_bin = [df[df["bin"] == b]["gc"].values for b in BIN_ORDER]

        bp = ax.boxplot(
            data_by_bin,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2,
                             solid_capstyle="round"),
            whiskerprops=dict(linewidth=1.0, color="black"),
            capprops=dict(linewidth=1.0, color="black"),
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor="gray",
                            markeredgecolor="none", alpha=0.4),
            boxprops=dict(linewidth=0.8, edgecolor="black"),
        )

        for patch, b in zip(bp["boxes"], BIN_ORDER):
            patch.set_facecolor(BIN_COLORS[b])
            patch.set_alpha(0.75)

        # 菱形标注均值（原版样式）
        for i, b in enumerate(BIN_ORDER):
            m = df[df["bin"] == b]["gc"].mean()
            ax.plot(i + 1, m, "D",
                    color="white", markeredgecolor="black",
                    markersize=6, zorder=5)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Strength bin")
        ax.set_ylabel("GC content")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"{name}   $r$ = {r:+.3f}")
        ax.grid(axis="y", color="#F0F0F0", linewidth=0.5, zorder=0)

        fig.tight_layout(pad=0.5)
        save_fig(fig, path_stem)


# ─────────────────────────────────────────
# Fig B/D — 散点图
# ─────────────────────────────────────────

def plot_scatter(df, name, r, path_stem):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

        plot_df = df.sample(n=min(3000, len(df)), random_state=42)
        sc = ax.scatter(
            plot_df["strength"], plot_df["gc"],
            c=plot_df["strength"], cmap="RdYlBu_r",
            s=6, alpha=0.30, linewidths=0,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.03, fraction=0.046)
        cbar.set_label("Strength")
        cbar.ax.tick_params(labelsize=7)

        x = df["strength"].values
        y = df["gc"].values
        slope, intercept, *_ = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, slope * xline + intercept,
                color="black", linewidth=1.4, linestyle="--")

        ax.set_xlabel("Promoter strength")
        ax.set_ylabel("GC content")
        ax.set_title(f"{name} — scatter")
        ax.grid(color="#F0F0F0", linewidth=0.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

        p_val = stats.pearsonr(x, y)[1]
        ax.text(
            0.05, 0.93,
            f"$r$ = {r:+.3f}\n$p$ = {p_val:.1e}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="#CCCCCC",
                      alpha=0.85, linewidth=0.6),
        )

        fig.tight_layout(pad=0.5)
        save_fig(fig, path_stem)


# ─────────────────────────────────────────
# 入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    ECOLI_PATH = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
    YEAST_PATH = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
    OUT_DIR    = "/home/yt/Code/DNA-Diffusion/lunwen/gc_bias"

    import os; os.makedirs(OUT_DIR, exist_ok=True)

    np.random.seed(42)

    print("Loading E. coli ...")
    df_ecoli = load_and_prepare(ECOLI_PATH)
    print("Loading S. cerevisiae ...")
    df_yeast = load_and_prepare(YEAST_PATH)

    r_ecoli, _ = analyze(df_ecoli, "E. coli")
    r_yeast, _ = analyze(df_yeast,  "S. cerevisiae")

    plot_boxplot(df_ecoli, "E. coli",        r_ecoli, f"{OUT_DIR}/FigA_ecoli_boxplot")
    plot_boxplot(df_yeast, "S. cerevisiae",  r_yeast, f"{OUT_DIR}/FigB_yeast_boxplot")
    plot_scatter(df_ecoli, "E. coli",        r_ecoli, f"{OUT_DIR}/FigC_ecoli_scatter")
    plot_scatter(df_yeast, "S. cerevisiae",  r_yeast, f"{OUT_DIR}/FigD_yeast_scatter")

    print("\nAll 8 files saved to:", OUT_DIR)