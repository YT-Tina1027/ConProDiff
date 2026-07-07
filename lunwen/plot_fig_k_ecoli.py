import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ── 路径 ────────────────────────────────────────────────────────
OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/tu3_ecoli"
FREQ_CSV = os.path.join(OUT_DIR, "motif_freq_table.csv")

# ── 统一样式（与第一/二份代码保持一致）──────────────────────────────
NATURE_RC = {
    "font.family": "sans-serif",
    "font.size":        11,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":   9,
    "legend.frameon":   False,
    "axes.linewidth":    0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
}

LABEL_ORDER = ["low", "mid", "high"]

# ── 与第二份代码完全一致的配色体系 ──────────────────────────────────
# low: 蓝色系  mid: 紫色系  high: 玫粉系
COND_PALETTE = {
    "low": {
        "fill_light": "#DCE9F5",
        "fill_mid":   "#ADC9E8",
        "fill_dark":  "#6FA8D4",
        "line":       "#4A86B8",
        "marker":     "#2E5F8A",
    },
    "mid": {
        "fill_light": "#E4DFF4",
        "fill_mid":   "#C0B4E4",
        "fill_dark":  "#8F7FCB",
        "line":       "#6B5BAD",
        "marker":     "#4A3A82",
    },
    "high": {
        "fill_light": "#F5DDE0",
        "fill_mid":   "#E8AABA",
        "fill_dark":  "#D4758A",
        "line":       "#B84D65",
        "marker":     "#8A2A42",
    },
}

LOW_COLOR  = COND_PALETTE["low"]["fill_dark"]   # "#6FA8D4" 蓝
MID_COLOR  = COND_PALETTE["mid"]["fill_dark"]   # "#8F7FCB" 紫
HIGH_COLOR = COND_PALETTE["high"]["fill_dark"]  # "#D4758A" 玫粉

# ── 读数据 ───────────────────────────────────────────────────────
freq_df = pd.read_csv(FREQ_CSV, index_col=0)[["low", "mid", "high"]]

# ── 三组均衡筛选 motif ──────────────────────────────────────────
def plot_fig_k(n_each: int = 10):
    diff = freq_df["high"] - freq_df["low"]

    high_idx = diff.nlargest(n_each).index
    low_idx  = diff.nsmallest(n_each).index

    used      = set(high_idx.tolist() + low_idx.tolist())
    remaining = diff.drop(index=list(used))
    sim_idx   = remaining.abs().nsmallest(n_each).index

    ordered_idx = list(diff.loc[
        list(high_idx) + list(low_idx) + list(sim_idx)
    ].sort_values(ascending=True).index)

    df       = freq_df.loc[ordered_idx].copy()
    diff_ord = diff.loc[ordered_idx]

    n = len(df)
    y = np.arange(n)

    # 条形图颜色：high-enriched → 玫粉，low-enriched → 蓝，similar → 灰
    HIGH_THRESH =  0.05
    LOW_THRESH  = -0.05
    bar_colors = []
    for d in diff_ord.values:
        if d > HIGH_THRESH:
            bar_colors.append(COND_PALETTE["high"]["fill_dark"])   # 玫粉
        elif d < LOW_THRESH:
            bar_colors.append(COND_PALETTE["low"]["fill_dark"])    # 蓝
        else:
            bar_colors.append("#BBBBBB")                           # 灰（中性）

    fig_w = 5.2
    fig_h = max(4.0, n * 0.15 + 1.2)

    with plt.rc_context(NATURE_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = gridspec.GridSpec(
            1, 2,
            width_ratios=[1.6, 1.0],
            wspace=0.10,
            left=0.28, right=0.96, top=0.88, bottom=0.10,
        )
        ax_dot = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        fig.suptitle(
            "Motif Enrichment Across Strength Levels",
            fontsize=11, x=0.62, y=0.96, ha="center",
        )

        # ══════════════════════════════════════════════════════════
        # 左图：哑铃图
        # ══════════════════════════════════════════════════════════
        low_v  = df["low"].values
        mid_v  = df["mid"].values
        high_v = df["high"].values

        for i in range(n):
            ax_dot.plot(
                [low_v[i], high_v[i]], [y[i], y[i]],
                color="#CCCCCC", lw=1.2, zorder=1, solid_capstyle="round",
            )

        ax_dot.scatter(mid_v, y, s=20, marker="D",
                       color=MID_COLOR, alpha=0.55, zorder=2, label="Mid")
        ax_dot.scatter(low_v, y, s=35, marker="o",
                       color=LOW_COLOR,  zorder=3, label="Low",
                       edgecolors="white", linewidths=0.5)
        ax_dot.scatter(high_v, y, s=35, marker="o",
                       color=HIGH_COLOR, zorder=3, label="High",
                       edgecolors="white", linewidths=0.5)

        ax_dot.set_yticks(y)
        ax_dot.set_yticklabels(df.index, fontsize=10)
        ax_dot.set_ylim(-0.8, n - 0.2)
        ax_dot.set_xlabel("Motif frequency in sequences", fontsize=11)

        x_max = max(df[["low", "mid", "high"]].values.max() * 1.08, 0.3)
        ax_dot.set_xlim(0, x_max)

        ax_dot.yaxis.grid(False)
        ax_dot.xaxis.grid(True, color="#EBEBF0", lw=0.5, ls="--", alpha=0.7, zorder=0)
        ax_dot.set_axisbelow(True)

        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=LOW_COLOR,
                   markersize=6, label="Low"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=MID_COLOR,
                   markersize=5, label="Mid", alpha=0.7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=HIGH_COLOR,
                   markersize=6, label="High"),
        ]
        ax_dot.legend(
            handles=legend_elems,
            title="Promoter\nstrength", title_fontsize=9,
            loc="lower right", fontsize=9,
            handletextpad=0.3, borderpad=0.5, labelspacing=0.3,
        )

        # ══════════════════════════════════════════════════════════
        # 右图：条形图
        # ══════════════════════════════════════════════════════════
        ax_bar.barh(
            y, diff_ord.values,
            height=0.60,
            color=bar_colors,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax_bar.axvline(0, color="#555555", lw=0.7, zorder=3)
        ax_bar.set_ylim(-0.8, n - 0.2)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Δ frequency\n(High − Low)", fontsize=11)
        ax_bar.set_title("Strength difference", fontsize=11, pad=6)

        max_abs = max(abs(diff_ord.values).max() * 1.12, 0.05)
        ax_bar.set_xlim(-max_abs, max_abs)

        ax_bar.xaxis.grid(True, color="#EBEBF0", lw=0.5, ls="--", alpha=0.7, zorder=0)
        ax_bar.set_axisbelow(True)
        ax_bar.spines["left"].set_visible(False)

        from matplotlib.patches import Patch
        bar_legend = [
            Patch(facecolor=COND_PALETTE["high"]["fill_dark"], alpha=0.88, label="High-enriched"),
            Patch(facecolor=COND_PALETTE["low"]["fill_dark"],  alpha=0.88, label="Low-enriched"),
            Patch(facecolor="#BBBBBB",                         alpha=0.88, label="Similar"),
        ]
        ax_bar.legend(
            handles=bar_legend, fontsize=9,
            loc="lower right",
            bbox_to_anchor=(1.5, 0.0),
            handlelength=0.9, handletextpad=0.4, borderpad=0.4, labelspacing=0.3,
        )

        for ext in ("pdf", "png"):
            out_path = os.path.join(OUT_DIR, f"FigK_dumbbell_strength.{ext}")
            fig.savefig(out_path, dpi=300)
            print(f"  Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Fig K …")
    plot_fig_k(n_each=10)
    print("Done.")