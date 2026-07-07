import os, sys, re, gc as _gc, warnings, logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shutil
shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)  # 清缓存，只需跑一次后可删
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

"""
酵母菌版 Fig A + Fig B
同步了大肠杆菌脚本的字体及色彩优化逻辑：
  Fig A — 各强度条件下预测活性分布箱线图（Real / Real(Oracle) / PromoDGDE / ConProDiff）
  Fig B — 混淆矩阵热图（ConProDiff vs PromoDGDE）
字体全局硬编码统一使用系统必备的 DejaVu Sans，确保 Linux 环境下绝对无警告弹窗。
"""

# ══════════════════════════════════════════════════════════════
# 0. Nature 主题 (彻底杜绝 findfont 警告，并消除隐形字体问题)
# ══════════════════════════════════════════════════════════════
NATURE_RC = {
    # ── 字体尺寸统一 12pt ──
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.frameon":     False,

    # ── 轴线与刻度 ──
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # ── 输出质量 ──
    "figure.dpi":     300,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
    "axes.unicode_minus": False,

    # ── 字体族 ──
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.weight":        "normal",
    "axes.titleweight":   "normal",
    "axes.labelweight":   "normal",
}


# ── Colorblind-safe 颜色 ──────────────────────────────────────
C_REAL  = "#7F9FB5"   # 深蓝灰  — Real
C_OUR   = "#E64B35"   # 红橙    — ConProDiff（our model）
C_PRO   = "#2E86C1"   # 青蓝    — PromoDGDE

# ── 尺寸 ──────────────────────────────────────────────────────
FIG_W = 3.54   # 每张图宽度（英寸）
FIG_H = 3.20   # 每张图高度（英寸）

LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

# ══════════════════════════════════════════════════════════════
# 1. 路径配置（按需修改）
# ══════════════════════════════════════════════════════════════
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/high.txt"

PROMODGDE_LOW_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/SC_80bp/low_final_sequences_SC.txt"
PROMODGDE_MID_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/SC_80bp/medium_final_sequences_SC.txt"
PROMODGDE_HIGH_TXT = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/SC_80bp/high_final_sequences_SC.txt"

REAL_CSV   = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
ORACLE_DIR = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_COND= "defined_media"

OUT_DIR    = "/home/yt/Code/DNA-Diffusion/lunwen/tu1_SC"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════
def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U","T") if c in "ACGT")

def read_txt(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    seqs = []
    for line in open(path):
        tok = max(re.split(r"[\s,\t;|]+", line.strip()), key=lambda x: len(standardize(x)), default="")
        s = standardize(tok)
        if len(s) >= 20: seqs.append(s)
    if not seqs: raise ValueError(f"No valid sequences in {path}")
    return seqs

def compute_thresholds(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next((c for c in ["strength","expression","exp"] if c in df.columns), None)
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(1/3)), float(vals.quantile(2/3))

def label_fn(x, q1, q2):
    if x <= q1: return "low"
    if x <= q2: return "mid"
    return "high"

def load_real_df(path, q1, q2):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    str_col = next((c for c in ["strength","expression","exp"] if c in df.columns), None)
    seq_col = next((c for c in ["sequence","seq","dna"]        if c in df.columns), None)
    df["sequence"] = df[seq_col].apply(standardize)
    df["strength"] = pd.to_numeric(df[str_col], errors="coerce")
    df = df.dropna(subset=["sequence","strength"])
    df = df[df["sequence"].str.len() > 0].copy()
    df["label"] = df["strength"].apply(lambda x: label_fn(x, q1, q2))
    return df

# ── Oracle ────────────────────────────────────────────────────
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

def condition_acc(scores, target_label, q1, q2, n_boot=500, seed=42):
    hits = np.array([label_fn(s, q1, q2) == target_label for s in scores], dtype=float)
    mean = hits.mean()
    rng  = np.random.RandomState(seed)
    boot = [rng.choice(hits, len(hits), replace=True).mean() for _ in range(n_boot)]
    ci   = np.std(boot) * 1.96
    return mean, ci

# ══════════════════════════════════════════════════════════════
# 3. 数据加载 & Oracle 打分
# ══════════════════════════════════════════════════════════════
print("="*58)
print(" Step 1 / 3 — Loading sequences")
print("="*58)

Q1, Q2 = compute_thresholds(REAL_CSV)
print(f"  Thresholds  Q1={Q1:.4f}  Q2={Q2:.4f}")

real_df = load_real_df(REAL_CSV, Q1, Q2)
print(f"  Real sequences loaded: {len(real_df)}")

gen_seqs = {
    "our":       {"low": read_txt(GEN_LOW_TXT), "mid": read_txt(GEN_MID_TXT), "high":read_txt(GEN_HIGH_TXT)},
    "PromoDGDE": {"low": read_txt(PROMODGDE_LOW_TXT), "mid": read_txt(PROMODGDE_MID_TXT), "high":read_txt(PROMODGDE_HIGH_TXT)},
}
for m, d in gen_seqs.items():
    for lbl, s in d.items():
        print(f"  {m:12s} | {lbl}: n={len(s)}")

print("\n Step 2 / 3 — Oracle scoring")
print("="*58)
g_tf, model_tf, scaler_tf, bs_tf = load_oracle()

real_scores = {lbl: real_df.loc[real_df["label"] == lbl, "strength"].dropna().values for lbl in LABEL_ORDER}

print(f"  Scoring Real (oracle) | n={len(real_df)} …", end=" ", flush=True)
real_oracle_all = oracle_score(real_df["sequence"].tolist(), g_tf, model_tf, scaler_tf, bs_tf)
real_df["oracle_pred"] = real_oracle_all

Q1_oracle = float(np.percentile(real_oracle_all, 33.3))
Q2_oracle = float(np.percentile(real_oracle_all, 66.7))
print(f"  Oracle Thresholds  Q1={Q1_oracle:.4f}  Q2={Q2_oracle:.4f}")

real_oracle_scores = {
    lbl: real_df.loc[real_df["oracle_pred"].apply(lambda x: label_fn(x, Q1_oracle, Q2_oracle)) == lbl, "oracle_pred"].dropna().values
    for lbl in LABEL_ORDER
}
print("done")

gen_scores = {"our": {}, "PromoDGDE": {}}
for method, seqs_by_lbl in gen_seqs.items():
    for lbl, seqs in seqs_by_lbl.items():
        print(f"  Scoring {method:12s} | {lbl} | n={len(seqs)} …", end=" ", flush=True)
        gen_scores[method][lbl] = oracle_score(seqs, g_tf, model_tf, scaler_tf, bs_tf)
        print("done")

# ══════════════════════════════════════════════════════════════
# 4. 条件准确率
# ══════════════════════════════════════════════════════════════
acc = {}
for method in ["our", "PromoDGDE"]:
    acc[method] = {}
    for lbl in LABEL_ORDER:
        sc = gen_scores[method][lbl]
        m_, ci_ = condition_acc(sc, lbl, Q1_oracle, Q2_oracle)
        acc[method][lbl] = (m_, ci_)
    vals = [acc[method][lbl][0] for lbl in LABEL_ORDER]
    acc[method]["overall"] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(3) * 1.96))

print("\nCondition accuracy:")
for method in ["our", "PromoDGDE"]:
    line = "  {:12s}  ".format(method)
    for lbl in LABEL_ORDER + ["overall"]:
        m_, ci_ = acc[method][lbl]
        line += f"{lbl}={m_:.3f}±{ci_:.3f}  "
    print(line)

# ══════════════════════════════════════════════════════════════
# 6. Fig A — 预测活性分布箱线图
# ══════════════════════════════════════════════════════════════
def plot_fig_a(save_stem="FigA_activity_boxplot_yeast"):
    C_REAL_ORC = "#1A5276"   

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        GROUP_POS = {"low": 1.0, "mid": 2.0, "high": 3.0}
        OFFSETS = {"Real": -0.33, "Real_Oracle": -0.11, "PromoDGDE": 0.11, "our": 0.33}
        COLORS = {"Real": C_REAL, "Real_Oracle": C_REAL_ORC, "our": C_OUR, "PromoDGDE": C_PRO}
        BOX_W = 0.16

        BP_KW = dict(widths=BOX_W, patch_artist=True, showfliers=False,
                     medianprops=dict(linewidth=1.6, solid_capstyle="round"),
                     whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))

        all_vals = []
        for lbl in LABEL_ORDER:
            data_map = {"Real": real_scores[lbl], "Real_Oracle": real_oracle_scores[lbl],
                        "our": gen_scores["our"][lbl], "PromoDGDE": gen_scores["PromoDGDE"][lbl]}
            for method, vals in data_map.items():
                if len(vals) == 0: continue

                x   = GROUP_POS[lbl] + OFFSETS[method]
                col = COLORS[method]

                bp = ax.boxplot(vals, positions=[x], **BP_KW,
                                boxprops=dict(facecolor=col, alpha=0.30, edgecolor=col, linewidth=0.9))
                bp["medians"][0].set_color(col)
                if method == "Real":
                    for patch in bp["boxes"]: patch.set_linestyle("--")

                n_j = min(len(vals), 60)
                idx = np.random.choice(len(vals), n_j, replace=False)
                jx  = np.random.normal(x, BOX_W * 0.18, n_j)
                ax.scatter(jx, vals[idx], s=3.0, alpha=0.20, color=col, edgecolors="none", zorder=2, rasterized=True)
                all_vals.extend(vals.tolist())

        lo_p = min(all_vals)
        hi_p = np.percentile(all_vals, 99.5)
        pad = (hi_p - lo_p) * 0.12
        ax.set_ylim(lo_p - pad * 0.5, hi_p + pad)

        ax.set_xticks([GROUP_POS[l] for l in LABEL_ORDER])
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_xlim(0.55, 3.45)
        ax.set_xlabel("Target strength condition")
        ax.set_ylabel("Expression score")

        legend_handles = [
            Patch(facecolor=C_REAL, edgecolor=C_REAL, alpha=0.70, linewidth=1.0, linestyle="--", label="Real (measured)"),
            Patch(facecolor=C_REAL_ORC, edgecolor=C_REAL_ORC, alpha=0.70, label="Real (oracle)"),
            Patch(facecolor=C_PRO, edgecolor=C_PRO, alpha=0.70, label="PromoDGDE"),
            Patch(facecolor=C_OUR, edgecolor=C_OUR, alpha=0.70, label="ConProDiff"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, 1.03),handlelength=1.2, handletextpad=0.5, borderpad=0.5, labelspacing=0.30)

        fig.tight_layout(pad=0.5)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f"  Saved {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 7. Fig B — 混淆矩阵（上：ConProDiff，下：PromoDGDE）
# ══════════════════════════════════════════════════════════════
def plot_fig_b(save_stem="FigB_confusion_matrix"):
    # ── 蓝色科技系渐变：极浅冰蓝 → 中钴蓝 → 深海军蓝 ──
    CMAP = LinearSegmentedColormap.from_list(
        "techblue",
        ["#F0F6FF", "#8DBDE8", "#3578C0"], 
        N=256,
    )
    methods = [("our", "ConProDiff"), ("PromoDGDE", "PromoDGDE")]

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(
            2, 1,
            figsize=(FIG_W + 0.8, FIG_H * 2 + 0.15),
        )

        for ax, (method_key, method_name) in zip(axes, methods):
            cm = np.zeros((3, 3), dtype=float)
            for i, target_lbl in enumerate(LABEL_ORDER):
                for s in gen_scores[method_key][target_lbl]:
                    pred_lbl = label_fn(s, Q1_oracle, Q2_oracle)
                    cm[i, LABEL_ORDER.index(pred_lbl)] += 1

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct   = np.where(row_sums > 0, cm / row_sums * 100, 0.0)
            im = ax.imshow(cm_pct, cmap=CMAP, vmin=0, vmax=100, aspect="auto")

            for i in range(3):
                for j in range(3):
                    val = cm_pct[i, j]
                    # 深色格子(>55%)用白色字，浅色格子用深蓝色字
                    text_color = "white" if val > 55 else "#0D2B4E"
                    ax.text(j, i, f"{val:.1f}%",
                            ha="center", va="center",
                            fontsize=12, color=text_color,)

            tick_labels = [LABEL_DISPLAY[l] for l in LABEL_ORDER]
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("Predicted condition", labelpad=4)
            ax.set_ylabel("Target condition", labelpad=4)
            ax.set_title(method_name, fontsize=12, pad=6, loc="center")

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
            cb.ax.tick_params(labelsize=12)
            cb.outline.set_linewidth(0.5)

        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"), bbox_inches=None)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 8. 运行
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n Step 3 / 3 — Plotting")
    print("="*58)
    np.random.seed(42)
    plot_fig_a()
    plot_fig_b()
    print("\nAll figures saved to:", OUT_DIR)