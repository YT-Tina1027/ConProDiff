"""
GC 条件嵌入消融实验 — 酵母菌 (S. cerevisiae)
与大肠杆菌版本（文档8）结构完全对齐，预测器替换为 TF Oracle

Fig1: GC 含量箱线图（Real / w/ GC / w/o GC，按低中高条件分组）
Fig2: 4-mer Pearson r 分组柱状图（w/ GC vs w/o GC）
Fig3: 条件准确率分组柱状图（w/ GC vs w/o GC，带 95% CI）
"""

import os, sys, gc, re, warnings, logging
import numpy as np
import pandas as pd
from itertools import product
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════
# 0. 样式（与大肠杆菌版完全对齐）
# ══════════════════════════════════════════════════════════════
NATURE_RC = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial"],
    "font.size":             9,       # 7 → 9
    "axes.labelsize":        9,       # 8 → 9
    "axes.titlesize":        9,       # 8 → 9
    "xtick.labelsize":       9,       # 7 → 9
    "ytick.labelsize":       9,       # 7 → 9
    "legend.fontsize":       8,
    "legend.frameon":        False,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.6,     # 0.7 → 0.6
    "ytick.major.width":     0.6,     # 0.7 → 0.6
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

W1, H1 = 3.6,3.0

# ── 配色（与大肠杆菌版完全对齐）────────────────────────────
C_REAL    = "#A3C4E0"
C_WITH    = "#DCA0AD"   # w/ GC  — 玫粉
C_WITHOUT = "#A99ED1"   # w/o GC — 紫

LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}
VARIANTS      = ["w/ GC", "w/o GC"]

# ══════════════════════════════════════════════════════════════
# 1. 路径配置
# ══════════════════════════════════════════════════════════════
VARIANT_DIRS = {
    "w/ GC":  "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0",
    "w/o GC": "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_nogc_cfg2.0",
}

REAL_CSV                = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
ORACLE_DIR              = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_MODEL_CONDITIONS = "defined_media"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/lunwen/ablation/gc_ablation_SC"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════
def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U", "T")
                   if c in "ACGT")

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

def save_fig(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(p)
        print(f"  Saved {stem}.{ext}")
    plt.close(fig)

# ── TF Oracle 预测 ───────────────────────────────────────────
def predict_with_oracle(sequences):
    import tensorflow as tf
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    from aux import load_model, evaluate_model

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    graph = tf.Graph()
    with graph.as_default():
        model, scaler, batch_size = load_model(ORACLE_MODEL_CONDITIONS)

    preds = evaluate_model(sequences, model, scaler, batch_size, graph)
    return np.asarray(preds, dtype=float).reshape(-1)

# ── 阈值 & 标签 ──────────────────────────────────────────────
def compute_thresholds_from_oracle(real_df):
    seqs  = real_df["sequence"].tolist()
    preds = predict_with_oracle(seqs)
    q1    = float(np.percentile(preds, 33.3))
    q2    = float(np.percentile(preds, 66.7))
    return preds, q1, q2

def label_fn(x, q1, q2):
    return "low" if x <= q1 else ("mid" if x <= q2 else "high")

def condition_acc(scores, target_lbl, q1, q2, n_boot=500, seed=42):
    hits = np.array([label_fn(s, q1, q2) == target_lbl
                     for s in scores], dtype=float)
    mean = hits.mean()
    rng  = np.random.RandomState(seed)
    boot = [rng.choice(hits, len(hits), replace=True).mean()
            for _ in range(n_boot)]
    return mean, float(np.std(boot) * 1.96)

# ── GC 含量 ───────────────────────────────────────────────────
def gc_content(s):
    return (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else np.nan

def gc_array(seqs):
    return np.array([gc_content(s) for s in seqs])

# ── k-mer Pearson ─────────────────────────────────────────────
def all_kmers(k):
    return ["".join(p) for p in product("ATCG", repeat=k)]

def kmer_freq_vector(seqs, k=4):
    vocab = all_kmers(k)
    cnt   = Counter()
    total = 0
    for s in seqs:
        s = standardize(s)
        if len(s) < k:
            continue
        for i in range(len(s) - k + 1):
            km = s[i:i+k]
            if set(km).issubset(set("ATCG")):
                cnt[km] += 1
                total   += 1
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([cnt[km] / total for km in vocab])

def safe_pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) \
        if np.std(x) > 0 and np.std(y) > 0 else np.nan

def kmer_corr(seqs, ref_freq, k=4):
    return safe_pearson(ref_freq, kmer_freq_vector(seqs, k))

# ══════════════════════════════════════════════════════════════
# 3. 数据加载 & Oracle 打分
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print(" Loading & scoring (SC / yeast) …")
print("=" * 60)

_real_df = pd.read_csv(REAL_CSV)
_real_df.columns = [c.strip().lower() for c in _real_df.columns]
_seq_col = next(c for c in ["sequence", "seq", "dna"] if c in _real_df.columns)
_real_df["sequence"] = _real_df[_seq_col].apply(standardize)
_real_df = _real_df[_real_df["sequence"].str.len() > 0].reset_index(drop=True)
print(f"  Real sequences: {len(_real_df)}")

print("  Scoring real sequences with Oracle …", end=" ", flush=True)
_real_oracle_preds, Q1, Q2 = compute_thresholds_from_oracle(_real_df)
print("done")
print(f"  Oracle thresholds  Q1={Q1:.4f}  Q2={Q2:.4f}")

_real_df["oracle_pred"] = _real_oracle_preds
_real_df["label"]       = [label_fn(s, Q1, Q2) for s in _real_oracle_preds]
real_gc   = {lbl: gc_array(_real_df.loc[_real_df["label"] == lbl, "sequence"].tolist())
             for lbl in LABEL_ORDER}
real_freq = {lbl: kmer_freq_vector(
                 _real_df.loc[_real_df["label"] == lbl, "sequence"].tolist())
             for lbl in LABEL_ORDER}
print("  Real GC & per-condition 4-mer freq computed")

gen_seqs, gen_scores = {}, {}
for var, dirpath in VARIANT_DIRS.items():
    gen_seqs[var], gen_scores[var] = {}, {}
    for lbl in LABEL_ORDER:
        path = os.path.join(dirpath, f"{lbl}.txt")
        seqs = read_txt(path)
        gen_seqs[var][lbl] = seqs
        print(f"  {var} | {lbl} n={len(seqs)} …", end=" ", flush=True)
        gen_scores[var][lbl] = predict_with_oracle(seqs)
        print("done")

# ══════════════════════════════════════════════════════════════
# 4. 指标计算
# ══════════════════════════════════════════════════════════════
acc, kmer_r = {}, {}
for var in VARIANTS:
    acc[var], kmer_r[var] = {}, {}
    for lbl in LABEL_ORDER:
        acc[var][lbl]    = condition_acc(gen_scores[var][lbl], lbl, Q1, Q2)
        kmer_r[var][lbl] = kmer_corr(gen_seqs[var][lbl], real_freq[lbl])

gen_gc = {var: {lbl: gc_array(gen_seqs[var][lbl]) for lbl in LABEL_ORDER}
          for var in VARIANTS}

print("\n" + "=" * 60)
print(f" {'Variant':>8}  {'Cond':>5}  {'Acc':>7}  {'±CI':>7}  {'4-mer r':>9}")
print("-" * 48)
for var in VARIANTS:
    for lbl in LABEL_ORDER:
        m_, ci_ = acc[var][lbl]
        kr_     = kmer_r[var][lbl]
        print(f" {var:>8}  {lbl:>5}  {m_:>7.4f}  {ci_:>7.4f}  {kr_:>9.4f}")
    print()

# ══════════════════════════════════════════════════════════════
# Fig 1 — GC 含量箱线图
# ══════════════════════════════════════════════════════════════
def plot_fig1(stem="Fig1_GC_boxplot_SC"):
    GROUP_X = {"low": 1.0, "mid": 2.0, "high": 3.0}
    OFFSETS = {"Real": -0.28, "w/ GC": 0.0, "w/o GC": 0.28}
    COLORS  = {"Real": C_REAL, "w/ GC": C_WITH, "w/o GC": C_WITHOUT}
    BOX_W   = 0.22

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(W1, H1))

        for lbl in LABEL_ORDER:
            data_map = {
                "Real":   real_gc[lbl],
                "w/ GC":  gen_gc["w/ GC"][lbl],
                "w/o GC": gen_gc["w/o GC"][lbl],
            }
            for src, vals in data_map.items():
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                xpos = GROUP_X[lbl] + OFFSETS[src]
                col  = COLORS[src]

                bp = ax.boxplot(
                    vals,
                    positions=[xpos],
                    widths=BOX_W,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=col, alpha=0.75,
                                  edgecolor=col, linewidth=0.8),
                    medianprops=dict(linewidth=1.6, color="white",
                                     solid_capstyle="round"),
                    whiskerprops=dict(linewidth=0.8, color=col),
                    capprops=dict(linewidth=0.8, color=col),
                )

                # ── 关键：patch_artist 模式下强制覆盖边框色 ──
                for patch in bp["boxes"]:
                    patch.set_edgecolor(col)

                n_j = min(len(vals), 80)
                idx = np.random.choice(len(vals), n_j, replace=False)
                jx  = np.random.normal(xpos, BOX_W * 0.15, n_j)
                ax.scatter(jx, vals[idx], s=2.5, alpha=0.25,
                           color=col, edgecolors="none", zorder=2,
                           rasterized=True)

        ax.set_xticks([GROUP_X[l] for l in LABEL_ORDER])
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_xlim(0.55, 3.45)
        # 2. 强行制造一个右轴，并塞入和图二几乎等宽的透明文字占位
        ax_fake = ax.twinx()
        ax_fake.set_ylabel(" ", color="none") # 注入一个透明的单位/空格
        ax_fake.set_yticklabels(["0.00", "0.20", "0.40", "0.60", "0.80", "1.00"], color="none") # 用图二的数字数字把空间撑开，但颜色设为透明
        ax_fake.spines["right"].set_linewidth(0.8)
        ax.set_xlabel("Target strength condition")
        ax.set_ylabel("GC content")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        legend_handles = [
            Patch(facecolor=C_REAL,    edgecolor=C_REAL,    alpha=0.75, label="Real"),
            Patch(facecolor=C_WITH,    edgecolor=C_WITH,    alpha=0.75, label="w/ GC"),
            Patch(facecolor=C_WITHOUT, edgecolor=C_WITHOUT, alpha=0.75, label="w/o GC"),
        ]
        # ── 图例改为左上角 ──
        ax.legend(handles=legend_handles, loc="upper left", frameon=False,
                  handlelength=1.0, handletextpad=0.4,
                  borderpad=0.5, labelspacing=0.3)

        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_linewidth(0.8)
        ax.spines["top"].set_color("black")
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_linewidth(0.8)
        ax.spines["right"].set_color("black")

        fig.subplots_adjust(left=0.18, right=0.92, bottom=0.15, top=0.88)
        save_fig(fig, stem)

# ══════════════════════════════════════════════════════════════
# Fig 2 — 4-mer r 分组柱状图 + 条件准确率折线（双轴合图）
# ══════════════════════════════════════════════════════════════
def plot_fig2(stem="Fig2_kmer_r_acc"):
    BW      = 0.28
    OFFSETS = {"w/ GC": -BW / 2, "w/o GC": BW / 2}
    # 柱颜色与大肠杆菌版对齐
    COLORS = {"w/ GC": "#DCA0AD", "w/o GC": "#A3C4E0"}       # w/o GC: 玫粉
    LINE_COLORS = {"w/ GC": "#B85A70", "w/o GC": "#4A7DB5"}
    LINE_MARKERS = {"w/ GC": "o",       "w/o GC": "s"}
    x = np.array([1.0, 2.0, 3.0])

    with plt.rc_context(NATURE_RC):
        fig, ax1 = plt.subplots(figsize=(W1, H1))
        ax2 = ax1.twinx()

        # ── 柱状图：4-mer Pearson r（左轴）───────────────────
        for var in VARIANTS:
            r_vals = [kmer_r[var][lbl] for lbl in LABEL_ORDER]
            xpos   = x + OFFSETS[var]
            ax1.bar(xpos, r_vals,
                    width=BW,
                    color=COLORS[var],
                    alpha=0.80,
                    edgecolor="white",
                    linewidth=0.5,
                    label=var,
                    zorder=3)
            for xi, rv in zip(xpos, r_vals):
                ax1.text(xi, rv + 0.0005,
                         f"{rv:.3f}",
                         ha="center", va="bottom",
                         fontsize=6.5, color="black")   # 5.5 → 6.5，与大肠杆菌版对齐

        # ── 折线：条件准确率（右轴）─────────────────────────
        for var in VARIANTS:
            m_vals  = [acc[var][lbl][0] for lbl in LABEL_ORDER]
            ci_vals = [acc[var][lbl][1] for lbl in LABEL_ORDER]
            ax2.plot(x, m_vals,
                     color=LINE_COLORS[var],
                     marker=LINE_MARKERS[var],
                     linewidth=1.4, markersize=4,
                     linestyle="--",
                     zorder=4,
                     label=f"{var} acc.")
            ax2.errorbar(x, m_vals, yerr=ci_vals,
                         fmt="none",
                         ecolor=LINE_COLORS[var],
                         elinewidth=0.8,
                         capsize=2.5, capthick=0.7,
                         zorder=5)
            for xi, mv, ci in zip(x, m_vals, ci_vals):
                ax2.text(xi, mv + ci + 0.018,          # 0.008 → 0.018，与大肠杆菌版对齐
                         f"{mv:.2f}",
                         ha="center", va="bottom",
                         fontsize=5.5, color=LINE_COLORS[var])

        # ── 左轴格式（4-mer r）──────────────────────────────
        all_r  = [kmer_r[v][l] for v in VARIANTS for l in LABEL_ORDER]
        r_lo   = min(all_r)
        margin = (1.0 - r_lo) * 0.08
        ax1.set_ylim(max(0.0, r_lo - margin), 1.0)    # 与大肠杆菌版对齐
        ax1.set_xticks(x)
        ax1.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax1.set_xlim(0.55, 3.45)
        ax1.set_xlabel("Target strength condition")
        ax1.set_ylabel("4-mer Pearson r")
        ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax1.grid(axis="y", color="#F0F0F0", linewidth=0.5, zorder=0)
        ax1.spines["top"].set_visible(False)

        # ── 右轴格式（条件准确率）───────────────────────────
        all_m  = [acc[v][l][0] for v in VARIANTS for l in LABEL_ORDER]
        all_ci = [acc[v][l][1] for v in VARIANTS for l in LABEL_ORDER]
        y_lo   = min(m - c for m, c in zip(all_m, all_ci))
        y_hi   = max(m + c for m, c in zip(all_m, all_ci))
        pad    = (y_hi - y_lo) * 0.35
        ax2.set_ylim(max(0.0, y_lo - pad), min(1.0, y_hi + pad * 3))  # 与大肠杆菌版对齐
        ax2.set_ylabel("Condition accuracy", color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax2.spines["top"].set_visible(False)

        # ── 图例合并 ─────────────────────────────────────────
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 1.02),          # 1.03 → 1.04，与大肠杆菌版对齐
                   ncol=2, frameon=False,
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

                        
        fig.subplots_adjust(left=0.18, right=0.92, bottom=0.15, top=0.88)
        save_fig(fig, stem)

# ══════════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════════
np.random.seed(42)
plot_fig1()
plot_fig2()
print("\nAll saved to:", OUT_DIR)