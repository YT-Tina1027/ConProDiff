"""
Mean ISM Logo — 低/中/高强度启动子的平均重要性图
每个条件对所有序列的 hyp 取平均，画成 sequence logo
三个条件各一行，叠在同一张图中对比
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

# ─────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────
NPY_OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/ism_npy_ecoli"
PLOT_OUT_DIR = "/home/yt/Code/DNA-Diffusion/lunwen/importance_logo_modisco_ecoli"
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

CORE_LEN    = 165
NT_ORDER    = ["A", "C", "G", "T"]
LABEL_ORDER_PLOT = ["high", "mid", "low"]  # 高在上，低在下
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

COLOR_SCHEME = {"A": "#107C41", "C": "#1F4E79", "G": "#F19020", "T": "#C00000"}

# 每个条件的配色（用于子图背景标注）
COND_COLOR = {"low": "#6FA8D4", "mid": "#8F7FCB", "high": "#6E73B5"}

FIG_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "axes.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
}

# ─────────────────────────────────────────
# 1. 加载并平均 ISM 分数
# ─────────────────────────────────────────
def hyp_to_display_batch(hyp, oh):
    """
    向量化版本：对每条序列每个位置，
    取参考碱基对应的"其余3个碱基hyp均值"作为显示分数
    hyp: (N, 165, 4) → display: (N, 165, 4)，非参考碱基位置为0
    """
    # 全部4个碱基hyp之和 - 参考碱基hyp = 其余3个之和，再除以3
    ref_idx  = np.argmax(oh, axis=-1)                     # (N, 165)
    total    = hyp.sum(axis=-1)                            # (N, 165)
    ref_val  = hyp[
        np.arange(hyp.shape[0])[:, None],
        np.arange(hyp.shape[1])[None, :],
        ref_idx
    ]                                                      # (N, 165)
    others_mean = (total - ref_val) / 3.0                  # (N, 165)

    display = np.zeros_like(hyp)
    display[
        np.arange(hyp.shape[0])[:, None],
        np.arange(hyp.shape[1])[None, :],
        ref_idx
    ] = others_mean
    return display


def load_mean_importance(label):
    prefix = os.path.join(NPY_OUT_DIR, label)
    hyp = np.load(f"{prefix}_hyp.npy")   # (N, 165, 4)
    oh  = np.load(f"{prefix}_oh.npy")    # (N, 165, 4)
    print(f"  [{label}] hyp shape: {hyp.shape}")

    display = hyp_to_display_batch(hyp, oh)   # (N, 165, 4)
    mean_contrib = display.mean(axis=0)        # (165, 4)
    return mean_contrib


# ─────────────────────────────────────────
# 2. 绘图：三行 logo 叠在一张图
# ─────────────────────────────────────────
def plot_mean_logos():
    data = {}
    for label in ["low", "mid", "high"]:
        data[label] = load_mean_importance(label)

    x_pos = np.arange(CORE_LEN)   # 0 ~ 164

    with plt.rc_context(FIG_RC):
        fig, axes = plt.subplots(
            3, 1,
            figsize=(11.0, 4.2),
            sharex=True,
            gridspec_kw={"hspace": 0.55},
        )

        for ax, label in zip(axes, LABEL_ORDER_PLOT):
            mat = data[label]   # (165, 4)

            # 全局 y 范围（用于统一三个子图的 y 轴，方便比较）
            ymax_global = max(d.max() for d in data.values())
            ymin_global = min(d.min() for d in data.values())
            margin = max(abs(ymax_global), abs(ymin_global)) * 0.20

            df = pd.DataFrame(mat, columns=NT_ORDER, index=x_pos)

            logo = logomaker.Logo(
                df, ax=ax,
                color_scheme=COLOR_SCHEME,
                vpad=0.01, width=0.85,
                flip_below=True,
                show_spines=False,
            )

            # y 轴范围统一
            y_lo = (ymin_global - margin) if ymin_global < -1e-5 else -abs(ymax_global) * 0.12
            y_hi = ymax_global + margin
            if abs(y_hi - y_lo) < 1e-6:
                y_lo, y_hi = -0.01, 0.01
            ax.set_ylim(y_lo, y_hi)

            # 零线
            ax.axhline(0, color="#555555", lw=0.5, zorder=2)

            # 条件标签（左侧黑色文字）
            ax.text(
                0.02, 0.95, LABEL_DISPLAY[label],  # <--- 将 0.88 改为 0.95（甚至 0.98，越接近 1 越靠上）
                transform=ax.transAxes,
                ha="left", va="top",               # 保持左对齐和顶部对齐
                fontsize=7.5, fontweight="bold",
                color="black",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1) 
            )

            # y 轴只显示 0 和最大值
            yticks = [0.0]
            if ymax_global > 0.01:
                yticks.append(round(ymax_global, 3))
            ax.set_yticks(yticks)
            ax.tick_params(axis="y", labelsize=5.5)

            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(False)
            ax.set_xlim(-0.5, CORE_LEN - 0.5)

        # x 轴只在最下方子图显示
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].set_xlabel("Position (bp)", fontsize=7)
        logo.style_xticks(anchor=0, spacing=20, fmt="%d")

        fig.text(
            0.07, 0.5, "Mean Importance Score",
            va="center", ha="center",
            rotation="vertical", fontsize=7,
        )

        fig.suptitle(
            "Mean ISM Importance Across Promoter Strength Levels",
            fontsize=8.5, y=0.98, x=0.55,
        )

        for ext in ("pdf", "png"):
            out = os.path.join(PLOT_OUT_DIR, f"mean_ism_logo_3cond.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    plot_mean_logos()