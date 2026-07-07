"""
fimo_nature_combined.py
=======================
从原始序列文件出发，完整计算 FIMO 富集分析，
并输出 Nature 投稿风格三张图（无星号标注）。

流程：
  1. 读取 low/mid/high 三组生成序列
  2. 解析 JASPAR MEME 文件
  3. 蒙特卡洛估算每个 motif 的分数阈值
  4. 批量扫描命中率
  5. Fisher 精确检验 + FDR 校正
  6. 输出 Nature 风格：火山图 / 柱状图 / 热图
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

warnings.filterwarnings("ignore")

# ─── 路径配置 ────────────────────────────────────────
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/high.txt"
JASPAR_MEME  = "/home/yt/Code/DNA-Diffusion/lunwen/JASPAR2026_Scerevisiae_CORE.meme"
OUT_DIR      = "/home/yt/Code/DNA-Diffusion/lunwen/fimo_enrichment"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 常量 ────────────────────────────────────────────
CORE_LEN             = 80
NT_ORDER             = ["A", "C", "G", "T"]
NT2IDX               = {nt: i for i, nt in enumerate(NT_ORDER)}
LABEL_ORDER          = ["low", "mid", "high"]
SCORE_PVALUE_THRESHOLD = 1e-4
MIN_HIT_RATE         = 0.05
FDR_METHOD           = "fdr_bh"

# ─── Nature 全局样式 ─────────────────────────────────
NATURE_RC = {
    "font.size":           7,
    "axes.labelsize":      7,
    "axes.titlesize":      8,
    "xtick.labelsize":     6.5,
    "ytick.labelsize":     6.5,
    "legend.fontsize":     6.5,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.major.size":    2.5,
    "ytick.major.size":    2.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.facecolor":      "white",
    "figure.facecolor":    "white",
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "axes.grid":           False,
}

COL_UP   = "#C0392B"
COL_DOWN = "#2471A3"
COL_NS   = "#BDC3C7"

GROUP_PAL = {
    "high": "#C0392B",
    "mid":  "#27AE60",
    "low":  "#2471A3",
}

# ════════════════════════════════════════════════════
# 1. 序列读取
# ════════════════════════════════════════════════════
def standardize_seq(s):
    s = str(s).upper().strip().replace("U", "T")
    return "".join(c for c in s if c in "ATCG")

def read_sequences(path):
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if len(seq) >= CORE_LEN:
                seqs.append(seq[:CORE_LEN])
    print(f"  Read {len(seqs):>5d} seqs  ← {os.path.basename(path)}")
    return seqs

# ════════════════════════════════════════════════════
# 2. JASPAR MEME 解析
# ════════════════════════════════════════════════════
def parse_jaspar_meme(path):
    motifs = []
    cur_name, cur_rows, in_matrix = None, [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if cur_name and cur_rows:
                    motifs.append(_finalize_motif(cur_name, cur_rows))
                parts    = line.split()
                cur_name = parts[2] if len(parts) > 2 else parts[1]
                cur_rows, in_matrix = [], False
            elif line.startswith("letter-probability matrix"):
                in_matrix = True
            elif in_matrix and line and line[0].isdigit():
                vals = list(map(float, line.split()))
                if len(vals) == 4:
                    cur_rows.append(vals)
    if cur_name and cur_rows:
        motifs.append(_finalize_motif(cur_name, cur_rows))
    print(f"  Parsed {len(motifs)} motifs from JASPAR MEME.")
    return motifs

def _finalize_motif(name, rows):
    pwm   = np.array(rows, dtype=np.float64)
    rs    = pwm.sum(axis=1, keepdims=True)
    pwm   = pwm / np.where(rs == 0, 1, rs)
    bg    = 0.25
    eps   = 1e-9
    lodds = np.log2((pwm + eps) / bg)
    return {
        "name":      name,
        "lodds":     lodds,
        "max_score": float(lodds.max(axis=1).sum()),
        "length":    len(rows),
    }

# ════════════════════════════════════════════════════
# 3. 分数阈值（蒙特卡洛）
# ════════════════════════════════════════════════════
def _build_score_threshold(motif, p_threshold=SCORE_PVALUE_THRESHOLD,
                            n_samples=50000):
    L     = motif["length"]
    lodds = motif["lodds"]
    rng   = np.random.default_rng(42)
    rand_seqs = rng.integers(0, 4, size=(n_samples, L))
    scores    = lodds[np.arange(L), rand_seqs].sum(axis=1)
    return float(np.quantile(scores, 1.0 - p_threshold))

# ════════════════════════════════════════════════════
# 4. 序列扫描
# ════════════════════════════════════════════════════
def seq_to_idx(seq):
    return np.array([NT2IDX.get(c, -1) for c in seq], dtype=np.int8)

def scan_sequence(seq_idx, motif_lodds, threshold, L):
    seq_len = len(seq_idx)
    if seq_len < L:
        return False
    for start in range(seq_len - L + 1):
        window = seq_idx[start:start+L]
        if -1 in window:
            continue
        if motif_lodds[np.arange(L), window].sum() >= threshold:
            return True
    rc_map = np.array([3, 2, 1, 0], dtype=np.int8)
    rc_idx = rc_map[seq_idx[::-1]]
    for start in range(seq_len - L + 1):
        window = rc_idx[start:start+L]
        if -1 in window:
            continue
        if motif_lodds[np.arange(L), window].sum() >= threshold:
            return True
    return False

def scan_group(seqs, motifs, thresholds):
    N, M         = len(seqs), len(motifs)
    hit_matrix   = np.zeros((N, M), dtype=bool)
    seq_idx_list = [seq_to_idx(s) for s in seqs]
    for j, (motif, thresh) in enumerate(zip(motifs, thresholds)):
        lodds = motif["lodds"]
        L     = motif["length"]
        for i, si in enumerate(seq_idx_list):
            hit_matrix[i, j] = scan_sequence(si, lodds, thresh, L)
        if (j + 1) % 50 == 0:
            print(f"    {j+1}/{M} motifs scanned...")
    return hit_matrix

# ════════════════════════════════════════════════════
# 5. Fisher 精确检验
# ════════════════════════════════════════════════════
def run_fisher(hit_A, hit_B, label_A, label_B):
    M       = hit_A.shape[1]
    results = []
    for j in range(M):
        a1 = hit_A[:, j].sum(); a0 = len(hit_A) - a1
        b1 = hit_B[:, j].sum(); b0 = len(hit_B) - b1
        odds, pval = fisher_exact([[a1, a0], [b1, b0]], alternative="two-sided")
        results.append({
            "motif_idx":           j,
            f"hit_rate_{label_A}": a1 / len(hit_A),
            f"hit_rate_{label_B}": b1 / len(hit_B),
            "odds_ratio":          odds,
            "pvalue":              pval,
        })
    df = pd.DataFrame(results)
    _, qvals, _, _ = multipletests(df["pvalue"].values, method=FDR_METHOD)
    df["qvalue"]  = qvals
    df["sig_fdr"] = qvals < 0.05
    return df

# ════════════════════════════════════════════════════
# 图1：Nature 风格火山图
# ════════════════════════════════════════════════════
def plot_volcano_nature(rate_df, fisher_hl, out_prefix, top_label=12):
    merged          = rate_df.copy()
    merged["qvalue"]  = fisher_hl["qvalue"].values
    merged["log2OR"]  = np.log2(fisher_hl["odds_ratio"].clip(1e-3, 1e3).values)
    merged["-logQ"]   = -np.log10(merged["qvalue"].clip(1e-300))
    merged            = merged.dropna()

    sig_up   = (merged["qvalue"] < 0.05) & (merged["log2OR"] > 0)
    sig_down = (merged["qvalue"] < 0.05) & (merged["log2OR"] < 0)
    ns       = ~(sig_up | sig_down)

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.2))

        ax.scatter(merged.loc[ns,       "log2OR"], merged.loc[ns,       "-logQ"],
                   s=8,  color=COL_NS,   alpha=0.6,  linewidths=0, zorder=1, rasterized=True)
        ax.scatter(merged.loc[sig_down, "log2OR"], merged.loc[sig_down, "-logQ"],
                   s=14, color=COL_DOWN, alpha=0.85, linewidths=0.3, edgecolors="white", zorder=3)
        ax.scatter(merged.loc[sig_up,   "log2OR"], merged.loc[sig_up,   "-logQ"],
                   s=14, color=COL_UP,   alpha=0.85, linewidths=0.3, edgecolors="white", zorder=3)

        ax.axhline(-np.log10(0.05), color="#7F8C8D", linewidth=0.7, linestyle="--", zorder=2)
        ax.axvline(0,               color="#95A5A6", linewidth=0.5,                  zorder=2)

        top_hits = merged[sig_up].nlargest(top_label, "-logQ")
        texts = []
        for name, row in top_hits.iterrows():
            t = ax.text(row["log2OR"], row["-logQ"], name,
                        fontsize=5.5, color="black",
                        fontweight="normal", ha="left", va="bottom")
            texts.append(t)

        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5),
            expand_points=(1.3, 1.5),
            force_text=(0.5, 0.8),
        )

        ax.set_xlabel("log\u2082(Odds Ratio)  [High / Low]", labelpad=4)
        ax.set_ylabel("\u2212log\u2081\u2080(\u03B1-adjusted \u03C1-value)", labelpad=4)

        handles = [
            mpatches.Patch(color=COL_UP,   label="High enriched (q<0.05)"),
            mpatches.Patch(color=COL_DOWN, label="Low enriched (q<0.05)"),
            mpatches.Patch(color=COL_NS,   label="Not significant"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=5.5, handlelength=1.0, handleheight=0.8)

        ax.set_xlim(merged["log2OR"].min() - 0.5, merged["log2OR"].max() + 1.2)
        ax.set_ylim(-1, merged["-logQ"].max() * 1.08)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}")
        plt.close(fig)
    print(f"  [Fig1] Saved → {out_prefix}.pdf/.png")

# ════════════════════════════════════════════════════
# 图2：Nature 风格柱状图（纵向，无星号）
# ════════════════════════════════════════════════════
def plot_barplot_nature(rate_df, fisher_hl, out_prefix, top_n=15):
    merged           = rate_df.copy()
    merged["qvalue"] = fisher_hl["qvalue"].values
    merged["log2OR"] = np.log2(fisher_hl["odds_ratio"].clip(1e-3, 1e3).values)

    sig = merged[(merged["qvalue"] < 0.05) & (merged["high"] > merged["low"])]
    sig = sig.nlargest(top_n, "log2OR")
    sig = sig.sort_values("high", ascending=False)   # 从左到右，命中率最高在左侧

    with plt.rc_context(NATURE_RC):
        n       = len(sig)
        bar_w   = 0.22
        gap     = 0.10
        unit    = bar_w * 3 + gap
        x_pos   = np.arange(n) * unit

        fig_w   = max(3.8, unit * n + 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, 3.2))

        ax.bar(x_pos + bar_w,  sig["high"].values, bar_w,
               color=GROUP_PAL["high"], label="High", zorder=3)
        ax.bar(x_pos,          sig["mid"].values,  bar_w,
               color=GROUP_PAL["mid"],  label="Mid",  zorder=3)
        ax.bar(x_pos - bar_w,  sig["low"].values,  bar_w,
               color=GROUP_PAL["low"],  label="Low",  zorder=3)

        y_max = max(sig["high"].max(), sig["mid"].max()) * 1.05

        # x 轴：motif 名称，45° 斜排防重叠
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sig.index, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel("Motif Hit Rate", labelpad=4)
        ax.set_ylim(0, y_max + 0.02)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # 横向辅助线
        for yg in ax.get_yticks():
            ax.axhline(yg, color="#E8E8E8", linewidth=0.4, zorder=0)

        ax.legend(loc="upper right", frameon=False,
                  fontsize=6, handlelength=1.0, handleheight=0.8, borderpad=0.3)

        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}")
        plt.close(fig)
    print(f"  [Fig2] Saved → {out_prefix}.pdf/.png")

# ════════════════════════════════════════════════════
# 图3：Nature 风格热图
# ════════════════════════════════════════════════════
def plot_heatmap_nature(rate_df, out_prefix, top_n=30):
    df         = rate_df.copy()
    df["range"] = df["high"] - df["low"]
    df          = df[df[["low", "mid", "high"]].max(axis=1) >= MIN_HIT_RATE]
    df          = df.nlargest(top_n, "range").drop(columns="range")
    df          = df.sort_values("high", ascending=False)

    colors_seq = ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    nat_cmap   = LinearSegmentedColormap.from_list("nat_blue", colors_seq, N=256)

    with plt.rc_context(NATURE_RC):
        n_rows  = len(df)
        fig, ax = plt.subplots(figsize=(2.6, n_rows * 0.26 + 0.9))

        vmax = min(df.values.max() * 1.05, 1.0)
        im   = ax.imshow(df.values, aspect="auto", cmap=nat_cmap,
                         vmin=0, vmax=vmax, interpolation="nearest")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(df.index, fontsize=6)
        ax.set_xlabel("Expression Group", labelpad=4)

        for i in range(n_rows):
            for j in range(3):
                val        = df.values[i, j]
                text_color = "white" if val > vmax * 0.6 else "#1A1A2E"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=5.2, color=text_color)

        for x in [-0.5, 0.5, 1.5, 2.5]:
            ax.axvline(x, color="white", linewidth=1.2)
        for y in np.arange(-0.5, n_rows, 1):
            ax.axhline(y, color="white", linewidth=0.5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, aspect=25)
        cbar.set_label("Hit Rate", fontsize=6.5, labelpad=3)
        cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
        cbar.outline.set_linewidth(0.4)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}")
        plt.close(fig)
    print(f"  [Fig3] Saved → {out_prefix}.pdf/.png")

# ════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("FIMO Enrichment → Nature-style figures")
    print("=" * 60)

    # 1. 读取序列
    print("\n[1/5] Loading sequences...")
    seqs = {label: read_sequences(path) for label, path in [
        ("low",  GEN_LOW_TXT),
        ("mid",  GEN_MID_TXT),
        ("high", GEN_HIGH_TXT),
    ]}

    # 2. 解析 JASPAR
    print("\n[2/5] Parsing JASPAR MEME...")
    motifs      = parse_jaspar_meme(JASPAR_MEME)
    motif_names = [m["name"] for m in motifs]
    M           = len(motifs)

    # 3. 计算阈值
    print(f"\n[3/5] Computing thresholds for {M} motifs (p={SCORE_PVALUE_THRESHOLD:.0e})...")
    thresholds = []
    for j, motif in enumerate(motifs):
        thresholds.append(_build_score_threshold(motif, SCORE_PVALUE_THRESHOLD))
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{M} done")
    print(f"  All {M} thresholds ready.")

    # 4. 批量扫描
    print("\n[4/5] Scanning sequences...")
    hit_matrices = {}
    for label in LABEL_ORDER:
        print(f"  [{label.upper()}] {len(seqs[label])} seqs × {M} motifs")
        hit_matrices[label] = scan_group(seqs[label], motifs, thresholds)
        rates = hit_matrices[label].mean(axis=0)
        print(f"    mean={rates.mean():.3f}  max={rates.max():.3f}  "
              f"motifs_with_hits={(rates>0).sum()}")

    # 5. Fisher 检验
    print("\n[5/5] Fisher exact tests (high vs low)...")
    fisher_hl = run_fisher(hit_matrices["high"], hit_matrices["low"], "high", "low")
    fisher_hl.insert(0, "motif_name", motif_names)
    fisher_hl = fisher_hl.set_index("motif_name")
    print(f"  Significant motifs (FDR<0.05): {fisher_hl['sig_fdr'].sum()}/{M}")

    # 构建命中率总表
    rate_df = pd.DataFrame(
        {label: hit_matrices[label].mean(axis=0) for label in LABEL_ORDER},
        index=motif_names
    )

    # 过滤
    mask            = (rate_df >= MIN_HIT_RATE).any(axis=1)
    rate_df_filtered = rate_df[mask]
    print(f"  Motifs passing filter (hit≥{MIN_HIT_RATE}): {mask.sum()}/{M}")

    # 保存 CSV
    rate_df.to_csv(os.path.join(OUT_DIR, "enrichment_matrix_all.csv"))
    fisher_hl.to_csv(os.path.join(OUT_DIR, "fisher_high_vs_low.csv"))
    print(f"  CSVs saved to {OUT_DIR}")

    # ── 绘图 ──────────────────────────────────────
    print("\nGenerating Nature-style figures...")

    fisher_hl_filtered = fisher_hl[fisher_hl.index.isin(rate_df_filtered.index)]

    print("[1/3] Volcano plot...")
    plot_volcano_nature(
        rate_df,
        fisher_hl,
        os.path.join(OUT_DIR, "fig1_volcano"),
        top_label=12
    )

    print("[2/3] Bar plot...")
    plot_barplot_nature(
        rate_df_filtered,
        fisher_hl_filtered,
        os.path.join(OUT_DIR, "fig2_barplot"),
        top_n=15
    )

    print("[3/3] Heatmap...")
    plot_heatmap_nature(
        rate_df_filtered,
        os.path.join(OUT_DIR, "fig3_heatmap"),
        top_n=30
    )

    # ── 摘要 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Top 15 motifs enriched in HIGH vs LOW (FDR sorted)")
    print("=" * 60)
    summary = fisher_hl.copy()
    summary["rate_low"]  = rate_df["low"]
    summary["rate_mid"]  = rate_df["mid"]
    summary["rate_high"] = rate_df["high"]
    top15 = summary.sort_values("qvalue").head(15)[
        ["rate_low", "rate_mid", "rate_high", "odds_ratio", "pvalue", "qvalue", "sig_fdr"]
    ]
    pd.set_option("display.float_format", "{:.4f}".format)
    print(top15.to_string())

    print(f"\nAll outputs saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()