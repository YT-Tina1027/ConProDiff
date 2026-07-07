"""
plot_motif_position_dist_SC.py
酵母（S. cerevisiae）Top5差异基序位置分布KDE图
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# ─────────────────────────────────────────
# ★ 路径配置（仅此处与ecoli版不同）
# ─────────────────────────────────────────
OUT_DIR   = "/home/yt/Code/DNA-Diffusion/lunwen/tu3_SC"
FREQ_CSV  = os.path.join(OUT_DIR, "motif_freq_table.csv")
STATS_PKL = os.path.join(OUT_DIR, "seq_stats.pkl")

SEQ_LEN   = 80    # ★ 酵母序列长度（ecoli是165）
TOP_N     = 5

COND_PALETTE = {
    "low":  {"line": "#4A86B8", "fill": "#DCE9F5"},
    "mid":  {"line": "#6B5BAD", "fill": "#E4DFF4"},
    "high": {"line": "#B84D65", "fill": "#F5DDE0"},
}
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}
LABEL_ORDER   = ["low", "mid", "high"]

NATURE_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    # "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
}

# ─────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────
freq_df = pd.read_csv(FREQ_CSV, index_col=0)[LABEL_ORDER]

with open(STATS_PKL, "rb") as f:
    stats = pickle.load(f)

motif_pos_all = stats["motif_positions"]

# ─────────────────────────────────────────
# 2. 选Top-N差异最大的基序
# ─────────────────────────────────────────
freq_std  = freq_df.std(axis=1).sort_values(ascending=False)
top_motifs = []
for motif in freq_std.index:
    counts = [len(motif_pos_all[lbl].get(motif, [])) for lbl in LABEL_ORDER]
    if max(counts) >= 10:
        top_motifs.append(motif)
    if len(top_motifs) == TOP_N:
        break

print(f"Top {TOP_N} motifs (S. cerevisiae):")
for m in top_motifs:
    row = freq_df.loc[m]
    print(f"  {m:20s}  low={row['low']:.3f}  mid={row['mid']:.3f}  high={row['high']:.3f}")

# ─────────────────────────────────────────
# 3. 绘图
# ─────────────────────────────────────────
x_bp = np.linspace(0, SEQ_LEN, 300)

with plt.rc_context(NATURE_RC):
    fig, axes = plt.subplots(
        1, TOP_N,
        figsize=(TOP_N * 2.8, 2.4),
        sharey=False,
        gridspec_kw={"wspace": 0.45},
    )

    if TOP_N == 1:
        axes = [axes]

    for ax, motif in zip(axes, top_motifs):
        for lbl in LABEL_ORDER:
            positions = motif_pos_all[lbl].get(motif, [])
            if len(positions) < 5:
                ax.axhline(0, color=COND_PALETTE[lbl]["line"],
                           lw=0.6, ls="--", alpha=0.4)
                continue

            pos_bp = np.array(positions) * SEQ_LEN
            kde    = gaussian_kde(pos_bp, bw_method="scott")
            y_kde  = kde(x_bp)

            ax.plot(x_bp, y_kde,
                    color=COND_PALETTE[lbl]["line"],
                    lw=1.6, label=LABEL_DISPLAY[lbl], zorder=3)

        ax.set_title(motif, fontsize=9, pad=4)
        ax.set_xlabel("Position (bp)", fontsize=8)
        ax.set_xlim(0, SEQ_LEN)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, SEQ_LEN // 2, SEQ_LEN])

        if ax is axes[0]:
            ax.set_ylabel("Density", fontsize=8)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        ax.get_legend().remove() if ax.get_legend() else None
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, ls="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

        # 删掉原来的 fig.suptitle 和 fig.legend，换成：

        fig.text(0.5, 1.02,
                 "Position Distribution of Top Differential TFBS Motifs",
                 ha="center", va="bottom", fontsize=10)

        legend_handles = [
            Line2D([0], [0], color=COND_PALETTE[lbl]["line"], lw=1.8,
                   label=LABEL_DISPLAY[lbl])
            for lbl in LABEL_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3, fontsize=9,
            handlelength=1.2, handletextpad=0.4,
            columnspacing=1.0, frameon=False,
            bbox_transform=fig.transFigure,   # ← 关键：锚定到figure坐标
        )

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"FigK_motif_position_dist_SC.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

print("Done.")