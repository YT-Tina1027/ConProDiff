import os
import sys
import re
import gc as _gc
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import product

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 1. 全局配置
# ══════════════════════════════════════════════════════════════
NATURE_RC = {
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],"font.weight": "normal",
    "axes.labelweight": "normal","axes.titleweight": "normal",
}
LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low Intensity", "mid": "Mid Intensity", "high": "High Intensity"}
COLOR_MAP     = {"low": "#4DBBD5", "mid": "#3C5488", "high": "#E64B35"}

# ══════════════════════════════════════════════════════════════
# 2. 路径配置
# ══════════════════════════════════════════════════════════════
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/high.txt"
REAL_CSV     = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
ORACLE_DIR   = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_COND  = "defined_media"
OUT_DIR      = "/home/yt/Code/DNA-Diffusion/lunwen/pearson_SC"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 3. 工具函数
# ══════════════════════════════════════════════════════════════
def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U","T") if c in "ACGT")

def read_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    seqs = []
    for line in open(path):
        tok = max(re.split(r"[\s,\t;|]+", line.strip()),
                  key=lambda x: len(standardize(x)), default="")
        s = standardize(tok)
        if len(s) >= 20:
            seqs.append(s)
    if not seqs:
        raise ValueError(f"No valid sequences in {path}")
    return seqs

def label_fn(x, q1, q2):
    return "low" if x <= q1 else ("mid" if x <= q2 else "high")

def load_real_df(path, q1, q2):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    str_col = next(c for c in ["strength","expression","exp"] if c in df.columns)
    seq_col = next(c for c in ["sequence","seq","dna"]       if c in df.columns)
    df["sequence"] = df[seq_col].apply(standardize)
    df["strength"] = pd.to_numeric(df[str_col], errors="coerce")
    df = df.dropna(subset=["sequence","strength"])
    df = df[df["sequence"].str.len() > 0].copy()
    df["label"] = df["strength"].apply(lambda x: label_fn(x, q1, q2))
    return df

def compute_thresholds(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next(c for c in ["strength","expression","exp"] if c in df.columns)
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(1/3)), float(vals.quantile(2/3))

def load_oracle():
    import tensorflow as tf
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    from aux import load_model
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    _gc.collect()
    g = tf.Graph()
    with g.as_default():
        model, scaler, bs = load_model(ORACLE_COND)
    return g, model, scaler, bs

def oracle_score(seqs, g, model, scaler, bs):
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    from aux import evaluate_model
    return np.asarray(evaluate_model(seqs, model, scaler, bs, g), dtype=float).reshape(-1)

# numpy 版本兼容：trapz 在新版改名 trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ══════════════════════════════════════════════════════════════
# 4. 数据加载
# ══════════════════════════════════════════════════════════════
print("Loading data and scoring with Oracle...")
Q1, Q2 = compute_thresholds(REAL_CSV)
real_df = load_real_df(REAL_CSV, Q1, Q2)

gen_seqs = {
    "low":  read_txt(GEN_LOW_TXT),
    "mid":  read_txt(GEN_MID_TXT),
    "high": read_txt(GEN_HIGH_TXT),
}

g_tf, model_tf, scaler_tf, bs_tf = load_oracle()

real_oracle_all = oracle_score(
    real_df["sequence"].tolist(), g_tf, model_tf, scaler_tf, bs_tf)
real_df["oracle_pred"] = real_oracle_all
Q1_oracle = float(np.percentile(real_oracle_all, 33.3))
Q2_oracle = float(np.percentile(real_oracle_all, 66.7))

gen_scores = {
    lbl: oracle_score(seqs, g_tf, model_tf, scaler_tf, bs_tf)
    for lbl, seqs in gen_seqs.items()
}

real_oracle_scores = {
    lbl: real_df.loc[
        real_df["oracle_pred"].apply(lambda x: label_fn(x, Q1_oracle, Q2_oracle)) == lbl,
        "oracle_pred"
    ].values
    for lbl in LABEL_ORDER
}

# ══════════════════════════════════════════════════════════════
# 5. k-mer 频率
# ══════════════════════════════════════════════════════════════
def compute_kmer_freq(seqs, k):
    bases = ['A','C','G','T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    counts = {km: 0 for km in kmers}
    total = 0
    for s in seqs:
        for i in range(len(s) - k + 1):
            sub = s[i:i+k]
            if sub in counts:
                counts[sub] += 1; total += 1
    freq = np.array([counts[km]/total if total > 0 else 0.0 for km in kmers])
    return kmers, freq

def build_kmer_df(kmer_lengths=[3,4,5]):
    rows = []
    for lbl in LABEL_ORDER:
        real_seqs_lbl = real_df.loc[real_df["label"]==lbl, "sequence"].tolist()
        gen_seqs_lbl  = gen_seqs[lbl]
        for k in kmer_lengths:
            kmers, real_freq = compute_kmer_freq(real_seqs_lbl, k)
            _,     gen_freq  = compute_kmer_freq(gen_seqs_lbl,  k)
            for km, rf, gf in zip(kmers, real_freq, gen_freq):
                rows.append({"strength_label":lbl,"k":k,"kmer":km,
                             "real_freq":rf,"gen_freq":gf})
    return pd.concat([pd.DataFrame([r]) for r in rows], ignore_index=True)

# ══════════════════════════════════════════════════════════════
# 6. 散点子函数
# ══════════════════════════════════════════════════════════════
def _draw_scatter(ax, x, y, color, rasterized=False):
    r, p = stats.pearsonr(x, y)
    mn, mx = min(x.min(),y.min()), max(x.max(),y.max())
    ax.plot([mn,mx],[mn,mx], color="#999999", linestyle="--", linewidth=0.6, zorder=1)
    ax.scatter(x, y, s=4, alpha=0.48, color=color, edgecolors="none", zorder=2,
               rasterized=rasterized)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.05, 0.95,
            f"$R^2={r**2:.3f}$\n$r={r:.3f}$\n{p_str}",
            transform=ax.transAxes, va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_title("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    return r, p

# ══════════════════════════════════════════════════════════════
# 7. Fig A：KDE 分布对比
#    用 fig.add_axes 以英寸为单位精确指定每个子图位置和尺寸，
#    三个子图物理宽高完全相同 → 视觉上等大。
#    不加 set_aspect（KDE x/y量纲不同）。
#    全局标题用 fig.text，坐标精确落在留白中央。
# ══════════════════════════════════════════════════════════════
def plot_intensity_distribution(
    real_scores_dict, gen_scores_dict,
    save_stem="FigA_intensity_distribution"
):
    from scipy.stats import gaussian_kde, ks_2samp

    # 物理尺寸（英寸）
    SUBPLOT_W = 1.55   # 每个子图宽
    SUBPLOT_H = 1.55   # 每个子图高（不加set_aspect，高度由此决定）
    N         = 3
    LEFT_IN   = 0.55   # 左侧留给y轴标题+刻度
    RIGHT_IN  = 0.15
    GAP_IN    = 0.45   # 子图间距
    BOT_IN    = 0.55   # 底部留给x轴标题+刻度
    TOP_IN    = 0.30

    FIG_W = LEFT_IN + N*SUBPLOT_W + (N-1)*GAP_IN + RIGHT_IN
    FIG_H = BOT_IN + SUBPLOT_H + TOP_IN

    ALPHA_REAL = 0.25
    ALPHA_GEN  = 0.10

    with plt.rc_context(NATURE_RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        axes = []
        for i in range(N):
            lf = (LEFT_IN + i*(SUBPLOT_W + GAP_IN)) / FIG_W
            bf = BOT_IN / FIG_H
            ax = fig.add_axes([lf, bf, SUBPLOT_W/FIG_W, SUBPLOT_H/FIG_H])
            axes.append(ax)

        for ax, lbl in zip(axes, LABEL_ORDER):
            color  = COLOR_MAP[lbl]
            r_vals = real_scores_dict[lbl]
            g_vals = gen_scores_dict[lbl]

            all_vals = np.concatenate([r_vals, g_vals])
            x_min, x_max = all_vals.min(), all_vals.max()
            margin = (x_max - x_min) * 0.10
            xs = np.linspace(x_min - margin, x_max + margin, 500)

            yr = gaussian_kde(r_vals, bw_method="scott")(xs)
            yg = gaussian_kde(g_vals, bw_method="scott")(xs)
            overlap = float(_trapz(np.minimum(yr, yg), xs))
            peak    = max(yr.max(), yg.max())
            yr_n = yr/peak; yg_n = yg/peak

            ax.fill_between(xs, yr_n, alpha=ALPHA_REAL, color=color, zorder=2)
            ax.plot(xs, yr_n, lw=1.2, color=color, linestyle="--", dashes=(4, 2.5), label="Real", zorder=3)
            ax.fill_between(xs, yg_n, alpha=ALPHA_GEN, color=color, zorder=1)
            ax.plot(xs, yg_n, lw=1.2, color=color, label="Generated", zorder=3)
            ax.axvline(np.mean(r_vals), color=color, lw=0.8, linestyle="--", alpha=0.55, zorder=4)
            ax.axvline(np.mean(g_vals), color=color, lw=0.8, linestyle="-",  alpha=0.55, zorder=4)
            ks_stat, ks_p = ks_2samp(r_vals, g_vals)
            p_str = "p < 0.001" if ks_p < 0.001 else f"p = {ks_p:.3f}"
            ax.text(0.97, 0.96,
                    f"KS = {ks_stat:.3f}\n{p_str}\nOVL = {overlap:.3f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=6.0,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.8))
            ax.fill_between(xs, np.minimum(yr_n,yg_n), alpha=0.30,
                            color="#888888", label="Overlap", zorder=2)

            leg = ax.legend(loc="upper left", fontsize=5.5, handlelength=2.0,
                      handletextpad=0.4, borderpad=0.3, labelspacing=0.3, frameon=False)
            for line in leg.get_lines():
                line.set_linewidth(1.0)
            ax.set_title(LABEL_DISPLAY[lbl], fontsize=7.5, pad=4)
            ax.set_ylim(-0.02, 1.12)
            ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0","0.5","1"])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.ticklabel_format(style="plain", axis="x", useOffset=False)
            ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
            if lbl != "low":
                ax.set_yticklabels([])

        # 全局标题：精确放在留白中央
        fig.text(0.5, (BOT_IN*0.45)/FIG_H,
                 "Oracle Predicted Score", ha="center", va="center", fontsize=8, fontweight="normal")
        fig.text((LEFT_IN*0.28)/FIG_W, 0.5,
                 "Density (normalized)",   ha="center", va="center",
                 fontsize=8, rotation="vertical", fontweight="normal")

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f" -> 已保存: {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 8. Fig B：k-mer 散点（正方形子图）
#    用 fig.add_axes 精确控制：
#    LPAD 足够宽 = 全局y标题宽(0.18) + 间距(0.05) + 行ylabel(0.45) + 刻度(0.35)
#    这样行标签和全局y标题之间有物理间隔，绝不重叠。
# ══════════════════════════════════════════════════════════════
def plot_kmer_correlation(kmer_df, kmer_lengths=[3,4,5], save_stem="FigB_kmer_correlation"):
    CELL  = 1.65   # 子图物理宽高（正方形）
    # 左侧分层：全局y标题占0.18in → 间距0.05in → 行ylabel占0.45in → 刻度/数字占0.35in
    LPAD  = 0.18 + 0.05 + 0.45 + 0.35   # = 1.03 in
    RPAD  = 0.10
    # 底部分层：全局x标题0.18 → 间距0.05 → 列xlabel0.35 → 刻度0.30
    BPAD  = 0.18 + 0.05 + 0.35 + 0.30   # = 0.88 in
    TPAD  = 0.10
    HGAP  = 0.45   # 行间距
    WGAP  = 0.40   # 列间距
    NROW, NCOL = 3, 3

    FIG_W = LPAD + NCOL*CELL + (NCOL-1)*WGAP + RPAD
    FIG_H = BPAD + NROW*CELL + (NROW-1)*HGAP + TPAD

    with plt.rc_context(NATURE_RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))

        for row_idx, lbl in enumerate(LABEL_ORDER):
            for col_idx, k in enumerate(kmer_lengths):
                r_from_bot = NROW - 1 - row_idx
                lf = (LPAD + col_idx*(CELL+WGAP)) / FIG_W
                bf = (BPAD + r_from_bot*(CELL+HGAP)) / FIG_H
                ax = fig.add_axes([lf, bf, CELL/FIG_W, CELL/FIG_H])

                sub = kmer_df[(kmer_df["strength_label"]==lbl) & (kmer_df["k"]==k)]
                if sub.empty:
                    continue
                _draw_scatter(ax, sub["gen_freq"].values, sub["real_freq"].values,
                              COLOR_MAP[lbl], rasterized=(k >= 5))
                ax.set_aspect(1.0, adjustable="box")
                ax.ticklabel_format(style='sci', axis='both',
                                    scilimits=(-2,-2), useOffset=False)
                ax.xaxis.get_offset_text().set_fontsize(6.5)
                ax.yaxis.get_offset_text().set_fontsize(6.5)

                if row_idx == NROW-1:
                    ax.set_xlabel(f"{k}-mer", labelpad=3, fontsize=8.5)
                else:
                    ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel(LABEL_DISPLAY[lbl], labelpad=4, fontsize=8.5)
                else:
                    ax.set_ylabel("")

        # 全局标题：x标题在BPAD留白中央，y标题在最左侧0.18in留白中央
        fig.text(0.5, (BPAD*0.38)/FIG_H,
                 "k-mer Frequency (Generated)", ha="center", va="center", fontsize=9, fontweight="normal")
        fig.text((LPAD * 0.45) / FIG_W, 0.5,
         "k-mer Frequency (Real)", ha="center", va="center",
         fontsize=9, rotation="vertical", fontweight="normal")

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f" -> 已保存: {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 9. 主程序
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("渲染 Fig A（分布对比）...")
    plot_intensity_distribution(real_oracle_scores, gen_scores)

    print("构建 Fig B k-mer 数据...")
    kmer_df = build_kmer_df(kmer_lengths=[3,4,5])

    print("渲染 Fig B...")
    plot_kmer_correlation(kmer_df)

    print("全部完成。")