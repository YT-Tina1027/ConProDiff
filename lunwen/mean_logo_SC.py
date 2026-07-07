import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

# ─────────────────────────────────────────
# 路径配置（酵母菌）
# ─────────────────────────────────────────
NPY_OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/ism_npy_SC"
PLOT_OUT_DIR = "/home/yt/Code/DNA-Diffusion/lunwen/importance_logo_modisco_SC"
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

CORE_LEN         = 80
NT_ORDER         = ["A", "C", "G", "T"]
LABEL_ORDER_PLOT = ["high", "mid", "low"]   # 高在上，低在下
LABEL_DISPLAY    = {"low": "Low", "mid": "Mid", "high": "High"}

COLOR_SCHEME = {"A": "#107C41", "C": "#1F4E79", "G": "#F19020", "T": "#C00000"}

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
# 1. 向量化 hyp_to_display + 均值
# ─────────────────────────────────────────
def hyp_to_display_batch(hyp, oh):
    ref_idx     = np.argmax(oh, axis=-1)          # (N, 80)
    total       = hyp.sum(axis=-1)                # (N, 80)
    ref_val     = hyp[
        np.arange(hyp.shape[0])[:, None],
        np.arange(hyp.shape[1])[None, :],
        ref_idx,
    ]                                             # (N, 80)
    others_mean = (total - ref_val) / 3.0         # (N, 80)

    display = np.zeros_like(hyp)
    display[
        np.arange(hyp.shape[0])[:, None],
        np.arange(hyp.shape[1])[None, :],
        ref_idx,
    ] = others_mean
    return display


def load_mean_importance(label):
    prefix = os.path.join(NPY_OUT_DIR, label)
    hyp = np.load(f"{prefix}_hyp.npy")   # (N, 80, 4)
    oh  = np.load(f"{prefix}_oh.npy")    # (N, 80, 4)
    print(f"  [{label}] hyp shape: {hyp.shape}")

    display      = hyp_to_display_batch(hyp, oh)   # (N, 80, 4)
    mean_contrib = display.mean(axis=0)             # (80, 4)
    return mean_contrib


# ─────────────────────────────────────────
# 2. 绘图（已修复坐标轴截断问题）
# ─────────────────────────────────────────
def plot_mean_logos():
    data = {}
    for label in ["low", "mid", "high"]:
        data[label] = load_mean_importance(label)

    x_pos = np.arange(CORE_LEN)

    with plt.rc_context(FIG_RC):
        fig, axes = plt.subplots(
            3, 1,
            figsize=(7.0, 4.5),   # 稍微增加了总高度，给独立的子图留出空间
            sharex=True,
            gridspec_kw={"hspace": 0.6}, # 增大子图间距，防止独立的y轴标签重叠
        )

        for ax, label in zip(axes, LABEL_ORDER_PLOT):
            mat = data[label]   # (80, 4)
            df  = pd.DataFrame(mat, columns=NT_ORDER, index=x_pos)

            # 1. 动态计算当前数据的最高与最低点
            # 因为 logomaker 会把正数和负数分别堆叠，所以要分别计算每一行的正数和与负数和
            pos_sum = df.clip(lower=0).sum(axis=1).max()
            neg_sum = df.clip(upper=0).sum(axis=1).min()
            
            # 2. 根据各自数据的最高/最低值设置 padding 比例（比如给上方和下方留出 15% 的安全空隙）
            y_hi = pos_sum * 1.15 if pos_sum > 1e-5 else 0.01
            y_lo = neg_sum * 1.15 if neg_sum < -1e-5 else -pos_sum * 0.1
            
            if abs(y_hi - y_lo) < 1e-6:
                y_lo, y_hi = -0.01, 0.01

            # 3. 渲染 Logo
            logo = logomaker.Logo(
                df, ax=ax,
                color_scheme=COLOR_SCHEME,
                vpad=0.005, width=0.85, # 适当减小了 vpad，让字母内部贴合更好
                flip_below=True,
                show_spines=False,
            )

            # 【已移除】删除了之前手动修改 Text 对象位置的代码，因为它会把最上面的字母推出轴外导致截断

            # 4. 严格将轴范围限制在安全计算值内
            ax.set_ylim(y_lo, y_hi)
            ax.axhline(0, color="#555555", lw=0.5, zorder=2)

            # 条件标签（黑色）
            ax.text(
                0.02, 1.07, LABEL_DISPLAY[label],  # <--- 相对于子图内部的坐标 (x=0.02, y=0.88)
                transform=ax.transAxes,
                ha="left", va="top",              # <--- 改为左对齐，顶部对齐
                fontsize=7.5, fontweight="bold",
                color="black",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1) # 可选：加个半透明白底，防止挡住后面的字母
            )

            # 每个子图根据自己的最大值展示独立的 y 轴刻度，避免刻度溢出
            yticks = [0.0]
            if pos_sum > 0.01:
                yticks.append(round(pos_sum, 3))
            ax.set_yticks(yticks)
            ax.tick_params(axis="y", labelsize=5.5)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(False)
            ax.set_xlim(-0.5, CORE_LEN - 0.5)

        # x 轴只在最下方显示
        axes[-1].spines["bottom"].set_visible(True)
        axes[-1].set_xlabel("Position (bp)", fontsize=7)
        logo.style_xticks(anchor=0, spacing=10, fmt="%d")

        fig.text(
            0.05, 0.5, "Mean Importance Score",
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