import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde

OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/tu3_ecoli"
FREQ_CSV = os.path.join(OUT_DIR, "motif_freq_table.csv")
REAL_FREQ_CSV = os.path.join(OUT_DIR, "motif_freq_table_real.csv")

# ========== 统一样式 ==========
NATURE_RC = {
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "legend.frameon": False, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
}
FIG_W, FIG_H  = 3.46, 3.10
LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

# ── 与第二份代码统一的配色体系 ──────────────────────────────────
# low: 蓝色系（对应 C_LOW "#A3C4E0"）
# mid: 紫色系（对应 C_MID "#A99ED1"）
# high: 玫粉系（对应 C_HIGH "#7B7FB8" → 改为与 C_OUR "#E8B8B8" 呼应的玫粉）
COND_PALETTE = {
    "low": {
        "fill_light": "#DCE9F5",   # 浅蓝
        "fill_mid":   "#ADC9E8",   # 中蓝
        "fill_dark":  "#6FA8D4",   # 深蓝（对应 C_LOW 系）
        "line":       "#4A86B8",
        "marker":     "#2E5F8A",
    },
    "mid": {
        "fill_light": "#E4DFF4",   # 浅紫
        "fill_mid":   "#C0B4E4",   # 中紫
        "fill_dark":  "#8F7FCB",   # 深紫（对应 C_MID 系）
        "line":       "#6B5BAD",
        "marker":     "#4A3A82",
    },
    "high": {
        "fill_light": "#F5DDE0",   # 浅玫粉（对应第二份 high 改色）
        "fill_mid":   "#E8AABA",   # 中玫粉
        "fill_dark":  "#D4758A",   # 深玫粉（对应 C_OUR "#E8B8B8" 深化）
        "line":       "#B84D65",
        "marker":     "#8A2A42",
    },
}
COND_COLORS = {lbl: p["fill_dark"] for lbl, p in COND_PALETTE.items()}

freq_df      = pd.read_csv(FREQ_CSV,      index_col=0)[["low", "mid", "high"]]
real_freq_df = pd.read_csv(REAL_FREQ_CSV, index_col=0)[["low", "mid", "high"]]
print(f"Loaded: {freq_df.shape[0]} motifs")

# =============================================================
# Fig G — 高富集 / 低富集 柱状图
# =============================================================
def plot_fig_g(top_n=10):
    diff = freq_df["high"] - freq_df["low"]
    activators = freq_df.loc[diff.nlargest(top_n).index].copy()
    repressors = freq_df.loc[diff.nsmallest(top_n).index].copy()
    min_step = 0.01

    activators = activators[
        (activators["low"] < activators["mid"]) &
        (activators["mid"] < activators["high"]) &
        (activators["mid"] - activators["low"] >= min_step) &
        (activators["high"] - activators["mid"] >= min_step)
    ].sort_values("high", ascending=False)

    repressors = repressors[
        (repressors["low"] > repressors["mid"]) &
        (repressors["mid"] > repressors["high"]) &
        (repressors["low"] - repressors["mid"] >= min_step) &
        (repressors["mid"] - repressors["high"] >= min_step)
    ].sort_values("low", ascending=False)

    for tag, df_sub, title in [
        ("G1", activators, "High-enriched motifs"),
        ("G2", repressors, "Low-enriched motifs"),
    ]:
        if len(df_sub) == 0:
            print(f"  No data for Fig{tag}, skipping")
            continue

        n = len(df_sub)
        fig_w = max(FIG_W, n * 0.45 + 1.0)

        with plt.rc_context(NATURE_RC):
            fig, ax = plt.subplots(figsize=(fig_w, FIG_H), constrained_layout=True)

            x     = np.arange(n)
            width = 0.25

            # 深色点（标注/边框色）与浅色填充，与第二份代码配色一致
            DOT_COLORS = {
                "low":  COND_PALETTE["low"]["fill_dark"],
                "mid":  COND_PALETTE["mid"]["fill_dark"],
                "high": COND_PALETTE["high"]["fill_dark"],
            }
            FILL_COLORS = {
                "low":  COND_PALETTE["low"]["fill_light"],
                "mid":  COND_PALETTE["mid"]["fill_light"],
                "high": COND_PALETTE["high"]["fill_mid"],
            }

            all_vals = df_sub[LABEL_ORDER].values.flatten()
            y_min_data = all_vals.min()
            y_max_data = all_vals.max()
            data_range = y_max_data - y_min_data

            y_bottom = max(0.0, y_min_data - data_range * 0.20)
            y_top    = y_max_data + data_range * 0.22

            for i, lbl in enumerate(LABEL_ORDER):
                vals = df_sub[lbl].values
                ax.bar(
                    x + (i - 1) * width, vals,
                    width=width,
                    color=FILL_COLORS[lbl],
                    alpha=0.65,
                    edgecolor=DOT_COLORS[lbl],
                    linewidth=0.8,
                    label=LABEL_DISPLAY[lbl],
                    zorder=3,
                    bottom=0,
                )

            ax.set_ylim(y_bottom, y_top)

            if y_bottom > 0:
                d = 0.012
                kwargs = dict(transform=ax.transAxes, color="k",
                              linewidth=0.8, clip_on=False)
                ax.plot((-d, +d), (-d*0.6, +d*0.6), **kwargs)
                ax.plot((-d, +d), (-d*0.6 + 0.022, +d*0.6 + 0.022), **kwargs)
                ax.spines["left"].set_bounds(y_bottom, y_top)

            ax.set_xticks(x)
            ax.set_xticklabels(df_sub.index, fontsize=9, rotation=35, ha="right")
            ax.set_ylabel("Fraction of sequences with motif", fontsize=11)
            ax.set_title(title, fontsize=11, pad=6)

            ax.yaxis.grid(True, color="#E0E0E8", linewidth=0.6,
                          linestyle="--", alpha=0.7, zorder=0)
            ax.set_axisbelow(True)

            ax.legend(
                title="Condition", title_fontsize=10,
                handlelength=1.0, handletextpad=0.4,
                borderpad=0.5, labelspacing=0.3, fontsize=9,
                loc="upper right",
            )

            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(OUT_DIR, f"Fig{tag}_bar.{ext}"), dpi=300)
            plt.close(fig)

        print(f"  Saved Fig{tag} ({n} TFs)")


# =============================================================
# Fig H — Top 12 variance 热图
# =============================================================
def plot_fig_h(top_n=12):
    top_motifs = (freq_df.std(axis=1)
                         .sort_values(ascending=False)
                         .head(top_n).index)
    df_top = freq_df.loc[top_motifs, LABEL_ORDER]

    # 使用第二份代码中 high 的玫粉系做渐变（与 C_OUR 系呼应）
    light = COND_PALETTE["low"]["fill_light"]    # "#DCE9F5"
    dark  = COND_PALETTE["low"]["fill_dark"]     # "#6FA8D4"
    cmap  = LinearSegmentedColormap.from_list("cmap_h", ["#ffffff", light, dark])

    fig_h = max(2.5, top_n * 0.38)
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W * 0.9, fig_h), constrained_layout=True)

        im = ax.imshow(df_top.values, aspect="auto", cmap=cmap,
                       vmin=0, vmax=1, interpolation="nearest")

        for row in range(top_n):
            for col in range(3):
                val = df_top.values[row, col]
                txt_color = "white" if val > 0.55 else "#333333"
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"], fontsize=10)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(df_top.index, fontsize=9)
        ax.set_ylabel("TF Motif", fontsize=9)
        ax.set_title(f"TFBS Enrichment Heatmap\n(Top {top_n} by cross-condition variance)",
                     fontsize=9)
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03, aspect=28)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Fraction of sequences with motif", fontsize=8)

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"FigH_heatmap_top{top_n}.{ext}"), dpi=300)
        plt.close(fig)
    print(f"  Saved FigH_heatmap_top{top_n}")


# =============================================================
# Fig J — Low vs High 散点图
# =============================================================
def plot_fig_j():
    from adjustText import adjust_text

    low_v  = freq_df["low"].values
    high_v = freq_df["high"].values
    names  = freq_df.index.tolist()
    eps    = 1e-4
    log2fc = np.log2((high_v + eps) / (low_v + eps))

    hi_mask    = log2fc > 0.5
    hi_indices = np.where(hi_mask)[0]
    hi_top     = hi_indices[np.argsort(log2fc[hi_indices])[::-1][:5]]

    lo_mask    = log2fc < -0.5
    lo_indices = np.where(lo_mask)[0]
    lo_top     = lo_indices[np.argsort(log2fc[lo_indices])[:5]]

    label_indices = set(hi_top.tolist() + lo_top.tolist())

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W * 1.2, FIG_W * 1.0), constrained_layout=True)

        neutral = (log2fc >= -0.5) & (log2fc <= 0.5)
        ax.scatter(low_v[neutral], high_v[neutral],
                   c="#CCCCCC", s=18, alpha=0.7, zorder=1, label="No change")
        # high-enriched → 玫粉（high 系，与第二份代码一致）
        ax.scatter(low_v[hi_mask], high_v[hi_mask],
                   c=COND_COLORS["high"], s=28, alpha=0.9, zorder=2,
                   label="High-enriched (log2FC > 0.5)")
        # low-enriched → 蓝色（low 系）
        ax.scatter(low_v[lo_mask], high_v[lo_mask],
                   c=COND_COLORS["low"], s=28, alpha=0.9, zorder=2,
                   label="Low-enriched (log2FC < −0.5)")

        lim = max(low_v.max(), high_v.max()) * 1.1
        ax.plot([0, lim], [0, lim], "--", color="#888888", lw=0.8, alpha=0.5)

        texts = []
        for i in label_indices:
            t = ax.text(low_v[i], high_v[i], names[i], fontsize=9, color="black")
            texts.append(t)

        adjust_text(
            texts,
            x=low_v, y=high_v,
            ax=ax,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black"),
            expand=(1.3, 1.5),
            force_text=(0.5, 0.8),
            force_points=(0.3, 0.5),
            max_move=50,
        )

        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Motif frequency — Low strength", fontsize=11)
        ax.set_ylabel("Motif frequency — High strength", fontsize=11)
        ax.set_title("Differential TFBS Enrichment:\nLow vs High Promoter Strength",
                     fontsize=11)
        ax.legend(fontsize=9, loc="lower right",
                  bbox_to_anchor=(1.05, 0.0),
                  handlelength=0.8, handletextpad=0.4, borderpad=0.5)

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"FigJ_scatter_low_vs_high.{ext}"), dpi=300)
        plt.close(fig)

    print("  Auto-labeled TFs:")
    print(f"    High-enriched: {[names[i] for i in hi_top]}")
    print(f"    Low-enriched:  {[names[i] for i in lo_top] if len(lo_top) else '(none)'}")
    print("  Saved FigJ_scatter_low_vs_high")


# =============================================================
# Fig I — Generated vs Real 小提琴图
# =============================================================
def plot_fig_i():
    with open(os.path.join(OUT_DIR, "seq_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    seq_n_motifs = stats["seq_n_motifs"]

    with open(os.path.join(OUT_DIR, "real_seq_stats.pkl"), "rb") as f:
        real_stats = pickle.load(f)
    real_n_motifs = real_stats["seq_n_motifs"]

    x_base = np.arange(len(LABEL_ORDER))
    offset = 0.2

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W * 1.2, FIG_W * 1.0), constrained_layout=True)

        for shift, source, n_motifs_dict, alpha_v, alpha_b in [
            (-offset, "Real",      real_n_motifs, 0.4, 0.65),
            ( offset, "Generated", seq_n_motifs,  0.6, 0.85),
        ]:
            positions = x_base + shift
            data_list = [n_motifs_dict[lbl] for lbl in LABEL_ORDER]

            parts = ax.violinplot(
                data_list, positions=positions,
                widths=0.32, showmedians=False, showextrema=False,
            )
            for lbl, pc in zip(LABEL_ORDER, parts["bodies"]):
                # Generated 用浅色，Real 用中色（与第二份代码风格一致）
                pc.set_facecolor(COND_PALETTE[lbl]["fill_light"] if source == "Generated"
                                 else COND_PALETTE[lbl]["fill_mid"])
                pc.set_edgecolor(COND_PALETTE[lbl]["line"])
                pc.set_linewidth(0.7)
                pc.set_alpha(alpha_v)

            bp = ax.boxplot(
                data_list, positions=positions, widths=0.13,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=1.2),
                whiskerprops=dict(linewidth=0.7),
                capprops=dict(linewidth=0.7),
            )
            for i, (lbl, patch) in enumerate(zip(LABEL_ORDER, bp["boxes"])):
                patch.set_facecolor(COND_COLORS[lbl])
                patch.set_alpha(alpha_b)
                if source == "Real":
                    patch.set_linestyle("--")
                    patch.set_linewidth(0.8)

                med = np.median(data_list[i])
                ax.text(positions[i], med + 0.15, f"{med:.0f}",
                        ha="center", va="bottom",
                        fontsize=9, color="white")

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#888888", alpha=0.5, linestyle="--",
                  linewidth=0.8, label="Real", edgecolor="#555555"),
            Patch(facecolor="#888888", alpha=0.85, label="Generated"),
        ]
        ax.legend(handles=legend_handles, fontsize=9,
                  handlelength=1.0, handletextpad=0.4, borderpad=0.5)
        ax.set_xticks(x_base)
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_xlabel("Promoter Strength Condition", fontsize=11)
        ax.set_ylabel("Unique TFBS motif types per sequence", fontsize=11)
        ax.set_title("Motif Diversity: Generated vs Real Sequences", fontsize=11)

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"FigI_gen_vs_real_violin.{ext}"), dpi=300)
        plt.close(fig)
    print("  Saved FigI_gen_vs_real_violin")


# ========== 运行 ==========
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Fig G …"); plot_fig_g()
    print("Fig H …"); plot_fig_h()
    print("Fig I …"); plot_fig_i()
    print("Fig J …"); plot_fig_j()
    print("\nAll figures saved to:", OUT_DIR)