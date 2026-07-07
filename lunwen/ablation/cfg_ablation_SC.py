"""
CFG Scale 消融实验 — 酵母菌（S. cerevisiae）
单张图：分组柱状图（准确率）+ 折线（4-mer Pearson r）
Oracle 打分器与 tu1_SC 脚本保持一致。
"""

import os, sys, re, gc as _gc, warnings, logging
import numpy as np
import pandas as pd
from itertools import product
from collections import Counter
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════
# 0. Nature 投稿样式
# ══════════════════════════════════════════════════════════════
NATURE_RC = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
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
    "axes.titleweight": "normal",
    "font.weight":      "normal",
    "axes.labelweight": "normal",
}

W1 = 3.54
H1 = 2.80

CFG_SCALES = ["1.0", "2.0", "3.0", "4.0"]

C_LOW  = "#A3C4E0"
C_MID  = "#A99ED1"
C_HIGH = "#DCA0AD"
COND_COLORS = {"low": C_LOW, "mid": C_MID, "high": C_HIGH}

LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

# ══════════════════════════════════════════════════════════════
# 1. 路径配置
# ══════════════════════════════════════════════════════════════
CFG_DIRS = {
    "1.0": "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg1.0",
    "2.0": "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0",
    "3.0": "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg3.0",
    "4.0": "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg4.0",
}
REAL_CSV   = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
ORACLE_DIR = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_COND= "defined_media"
OUT_DIR    = "/home/yt/Code/DNA-Diffusion/lunwen/ablation/cfg_ablation_SC"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════
def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U", "T")
                   if c in "ACGT")

def read_txt(path):
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
    return np.asarray(evaluate_model(seqs, model, scaler, bs, g),
                      dtype=float).reshape(-1)

def condition_acc(scores, target_lbl, q1, q2, n_boot=500, seed=42):
    hits = np.array([label_fn(s, q1, q2) == target_lbl
                     for s in scores], dtype=float)
    mean = hits.mean()
    rng  = np.random.RandomState(seed)
    boot = [rng.choice(hits, len(hits), replace=True).mean()
            for _ in range(n_boot)]
    return mean, float(np.std(boot) * 1.96)

def kmer_freq(seqs, k=4, trunc=100):
    keys = ["".join(p) for p in product("ACGT", repeat=k)]
    cnt  = Counter()
    for s in seqs:
        s = s[:trunc]
        for i in range(len(s) - k + 1):
            cnt[s[i:i+k]] += 1
    total = sum(cnt.values()) or 1
    return np.array([cnt[key] / total for key in keys])

def kmer_corr(seqs, ref_freq):
    r, _ = pearsonr(ref_freq, kmer_freq(seqs))
    return float(r)

def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p)
        print(f"  Saved {stem}.{ext}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 3. 数据加载 & Oracle 打分
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print(" Loading & scoring …")
print("=" * 60)

# 读取真实数据，计算 Oracle 阈值
_real_df = pd.read_csv(REAL_CSV)
_real_df.columns = [c.strip().lower() for c in _real_df.columns]
_seq_col = next(c for c in ["sequence", "seq", "dna"]        if c in _real_df.columns)
_str_col = next(c for c in ["strength", "expression", "exp"] if c in _real_df.columns)
_real_df["sequence"] = _real_df[_seq_col].apply(standardize)
_real_df["strength"] = pd.to_numeric(_real_df[_str_col], errors="coerce")
_real_df = _real_df.dropna(subset=["sequence", "strength"])
_real_df = _real_df[_real_df["sequence"].str.len() > 0].copy()
print(f"  Real sequences loaded: {len(_real_df)}")

# 加载 Oracle
g_tf, model_tf, scaler_tf, bs_tf = load_oracle()
print("  Oracle loaded")

# 用 Oracle 对真实序列打分，用预测值定阈值（与 tu1_SC 保持一致）
print(f"  Oracle scoring real seqs n={len(_real_df)} …", end=" ", flush=True)
_real_oracle = oracle_score(
    _real_df["sequence"].tolist(), g_tf, model_tf, scaler_tf, bs_tf)
Q1 = float(np.percentile(_real_oracle, 33.3))
Q2 = float(np.percentile(_real_oracle, 66.7))
print(f"done\n  Oracle thresholds  Q1={Q1:.4f}  Q2={Q2:.4f}")

# 真实序列 4-mer 参考频率
_real_freq = kmer_freq(_real_df["sequence"].tolist())
print("  4-mer reference freq computed")

# 加载 & 打分所有 CFG scale
gen_seqs, gen_scores = {}, {}
for scale, dirpath in CFG_DIRS.items():
    gen_seqs[scale], gen_scores[scale] = {}, {}
    for lbl in LABEL_ORDER:
        path = os.path.join(dirpath, f"{lbl}.txt")
        seqs = read_txt(path)
        gen_seqs[scale][lbl] = seqs
        print(f"  CFG {scale} | {lbl} n={len(seqs)} …", end=" ", flush=True)
        gen_scores[scale][lbl] = oracle_score(
            seqs, g_tf, model_tf, scaler_tf, bs_tf)
        print("done")

# ══════════════════════════════════════════════════════════════
# 4. 计算准确率 & 4-mer 相关性
# ══════════════════════════════════════════════════════════════
acc, kmer_r = {}, {}
for scale in CFG_SCALES:
    acc[scale], kmer_r[scale] = {}, {}
    for lbl in LABEL_ORDER:
        acc[scale][lbl]    = condition_acc(
            gen_scores[scale][lbl], lbl, Q1, Q2)
        kmer_r[scale][lbl] = kmer_corr(gen_seqs[scale][lbl], _real_freq)

    # overall 准确率：加权均值 + bootstrap CI
    ns      = {lbl: len(gen_seqs[scale][lbl]) for lbl in LABEL_ORDER}
    total_n = sum(ns.values())
    w_mean  = sum(acc[scale][lbl][0] * ns[lbl] for lbl in LABEL_ORDER) / total_n
    all_hits = np.concatenate([
        np.array([label_fn(s, Q1, Q2) == lbl
                  for s in gen_scores[scale][lbl]], dtype=float)
        for lbl in LABEL_ORDER
    ])
    rng  = np.random.RandomState(42)
    boot = [rng.choice(all_hits, len(all_hits), replace=True).mean()
            for _ in range(500)]
    acc[scale]["overall"] = (w_mean, float(np.std(boot) * 1.96))

    # overall 4-mer：拼合所有生成序列整体算
    all_seqs = [s for lbl in LABEL_ORDER for s in gen_seqs[scale][lbl]]
    kmer_r[scale]["overall"] = kmer_corr(all_seqs, _real_freq)

print("\nAccuracy summary:")
for scale in CFG_SCALES:
    parts = "  ".join(f"{l}={acc[scale][l][0]:.3f}" for l in LABEL_ORDER + ["overall"])
    print(f"  CFG {scale}: {parts}")

print("\n4-mer Pearson r:")
for scale in CFG_SCALES:
    parts = "  ".join(f"{l}={kmer_r[scale][l]:.4f}" for l in LABEL_ORDER + ["overall"])
    print(f"  CFG {scale}: {parts}")

# ══════════════════════════════════════════════════════════════
# 5. 绘图
# ══════════════════════════════════════════════════════════════
def plot_cfg_ablation(stem="FigAbl_cfg_ablation_SC"):
    x       = np.arange(len(CFG_SCALES))
    bw      = 0.20
    offsets = np.array([-1, 0, 1]) * bw

    with plt.rc_context(NATURE_RC):
        fig, ax1 = plt.subplots(figsize=(W1, H1))
        ax2 = ax1.twinx()

        # ── 柱状图：准确率 ───────────────────────────────────
        for i, (lbl, col) in enumerate(COND_COLORS.items()):
            means = [acc[s][lbl][0] for s in CFG_SCALES]
            cis   = [acc[s][lbl][1] for s in CFG_SCALES]
            ax1.bar(x + offsets[i], means, bw,
                    color=col, alpha=0.85,
                    edgecolor=col, linewidth=0.8,
                    label=LABEL_DISPLAY[lbl], zorder=3)
            ax1.errorbar(x + offsets[i], means, yerr=cis,
                         fmt="none",
                         ecolor="#555555",
                         elinewidth=0.8,
                         capsize=2.5,
                         capthick=0.7,
                         zorder=4)

        # ── 折线：overall 4-mer 相关性 ───────────────────────
        corr_vals = [kmer_r[s]["overall"] for s in CFG_SCALES]
        ax2.plot(x, corr_vals,
                 color="black", marker="D",
                 linewidth=1.4, markersize=4.5,
                 linestyle="--", zorder=4, label="4-mer corr.")

        # ── 数值标注（ylim 必须能容纳文字）──────────────────
        cspan       = max(corr_vals) - min(corr_vals)
        text_offset = (cspan * 0.15 if cspan > 1e-4
                       else abs(np.mean(corr_vals)) * 5e-5 + 1e-5)
        for xi, v in zip(x, corr_vals):
            ax2.text(xi, v + text_offset,
                     f"{v:.4f}", ha="center", fontsize=8, color="black")

        # ── 轴格式 ──────────────────────────────────────────
        ax1.set_xticks(x)
        ax1.set_xticklabels(CFG_SCALES)
        ax1.set_xlabel("CFG guidance scale")
        ax1.set_ylabel("Condition accuracy")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax1.set_ylim(0, 1.05)
        ax1.spines["top"].set_visible(False)

        ax2.set_ylabel("4-mer Pearson r", color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
        ax2.spines["top"].set_visible(False)
        y_low  = min(corr_vals) - max(cspan * 0.8, text_offset * 3)
        y_high = max(corr_vals) + text_offset * 8
        ax2.set_ylim(y_low, y_high)

        # ── 图例合并 ─────────────────────────────────────────
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2,
                   loc="upper left", ncol=2,
                   handlelength=1.0, handletextpad=0.3,
                   borderpad=0.4, labelspacing=0.2,
                   columnspacing=0.6)

        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(0.8)
        ax2.spines["right"].set_color("black")

        ax1.spines["top"].set_visible(True)
        ax1.spines["top"].set_linewidth(0.8)
        ax1.spines["top"].set_color("black")
        ax2.spines["top"].set_visible(True)
        ax2.spines["top"].set_linewidth(0.8)
        ax2.spines["top"].set_color("black")
        ax2.tick_params(axis="y", right=True, direction="out",
                width=0.6, length=3, labelcolor="black")

        fig.tight_layout(pad=0.5)
        save_fig(fig, stem)


np.random.seed(42)
plot_cfg_ablation()
print("\nSaved to:", OUT_DIR)

# ══════════════════════════════════════════════════════════════
# 6. 汇总统计表
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f" {'CFG':>4}  {'Cond':>7}  {'Acc':>7}  {'±CI':>7}  {'4-mer r':>10}")
print("-" * 48)
for scale in CFG_SCALES:
    for lbl in LABEL_ORDER + ["overall"]:
        m_, ci_ = acc[scale][lbl]
        kr_     = kmer_r[scale][lbl]
        print(f" {scale:>4}  {lbl:>7}  {m_:>7.4f}  {ci_:>7.4f}  {kr_:>10.4f}")
    print()