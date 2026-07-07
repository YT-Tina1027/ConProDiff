"""
plot_gen_vs_real_freq_SC.py
酵母（S. cerevisiae）生成序列 vs 真实序列的基序使用频率散点图
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ─────────────────────────────────────────
# ★ 路径配置（仅此处与ecoli版不同）
# ─────────────────────────────────────────
OUT_DIR       = "/home/yt/Code/DNA-Diffusion/lunwen/tu3_SC"
FREQ_CSV      = os.path.join(OUT_DIR, "motif_freq_table.csv")
REAL_FREQ_CSV = os.path.join(OUT_DIR, "motif_freq_table_real.csv")

LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

COND_PALETTE = {
    "low":  {"dot": "#6FA8D4", "line": "#4A86B8"},
    "mid":  {"dot": "#8F7FCB", "line": "#6B5BAD"},
    "high": {"dot": "#D4758A", "line": "#B84D65"},
}

NATURE_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
}

# ─────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────
gen_df  = pd.read_csv(FREQ_CSV,      index_col=0)[LABEL_ORDER]
real_df = pd.read_csv(REAL_FREQ_CSV, index_col=0)[LABEL_ORDER]

common_motifs = gen_df.index.intersection(real_df.index)
gen_df  = gen_df.loc[common_motifs]
real_df = real_df.loc[common_motifs]
print(f"共有基序数: {len(common_motifs)}")

# ─────────────────────────────────────────
# 2. 绘图
# ─────────────────────────────────────────
with plt.rc_context(NATURE_RC):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(9.0, 2.2),
        gridspec_kw={"wspace": 0.38},
    )

    for ax, lbl in zip(axes, LABEL_ORDER):
        x = real_df[lbl].values
        y = gen_df[lbl].values

        r, pval = pearsonr(x, y)
        p_str = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"

        ax.scatter(
            x, y,
            c=COND_PALETTE[lbl]["dot"],
            s=18, alpha=0.7,
            edgecolors="none",
            zorder=3,
        )

        lim = max(x.max(), y.max()) * 1.12
        ax.plot([0, lim], [0, lim],
                color="#AAAAAA", lw=0.9, ls="--", zorder=2)

        coef  = np.polyfit(x, y, 1)
        x_fit = np.linspace(0, lim, 200)
        y_fit = np.polyval(coef, x_fit)
        ax.plot(x_fit, y_fit,
                color=COND_PALETTE[lbl]["line"], lw=1.4, zorder=4)

        ax.text(
            0.05, 0.95,
            f"r = {r:.3f}\n{p_str}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.5,
            color=COND_PALETTE[lbl]["line"],
        )

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(LABEL_DISPLAY[lbl], fontsize=10, pad=5)
        ax.set_xlabel("Real sequences (motif frequency)", fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("Generated sequences\n(motif frequency)", fontsize=8.5)

        ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, ls="--", alpha=0.7, zorder=0)
        ax.xaxis.grid(True, color="#E8E8E8", lw=0.5, ls="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Generated vs Real Promoter Sequences: TFBS Motif Frequency Correlation",
        fontsize=10, y=1.03,
    )

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"FigK_gen_vs_real_freq_SC.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

print("Done.")