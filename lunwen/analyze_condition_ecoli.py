import os
import sys
import re
import warnings
import logging
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
import torch
from sklearn.preprocessing import StandardScaler

# 将预测器路径加入系统路径并导入模型
PREDICTOR_DIR = "/home/yt/Code/DNA-Diffusion/Predictor"
if PREDICTOR_DIR not in sys.path:
    sys.path.insert(0, PREDICTOR_DIR)
from predictor_models import LSTMModel

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

"""
大肠杆菌版 Fig A + Fig B
与酵母菌脚本（文档3）结构完全对齐：
  Fig A — 各强度条件下预测活性分布箱线图（Real实测 / Real(LSTM预测) / PromoDGDE / ConProDiff）
  Fig B — 混淆矩阵（PromoDGDE vs ConProDiff）
主要差异：用 LSTM predictor 替换 TF Oracle
配色：蓝色科技系 + 琥珀色强调色；字体统一 12pt
"""

# ══════════════════════════════════════════════════════════════
# 0. 配置与配置参数 (Nature 主题，字体统一 12pt，蓝色科技配色)
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
}

# ── 蓝色科技系配色 ────────────────────────────────────────────
# Real (measured)  : 钢铁蓝灰，低饱和，表示"基准"
C_REAL      = "#7F9FB5"
# Real (LSTM)      : 深海军蓝，高权威感
C_REAL_ORC  = "#1A5276"
# PromoDGDE        : 科技亮蓝，中等强度
C_PRO       = "#2E86C1"
# ConProDiff (ours): 琥珀橙，与蓝色系形成最大对比，突出"我们的方法"
C_OUR       = "#E64B35"
# ─────────────────────────────────────────────────────────────

FIG_W, FIG_H = 3.54, 3.20
LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

# ══════════════════════════════════════════════════════════════
# 1. 路径配置
# ══════════════════════════════════════════════════════════════
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/high.txt"

PROMODGDE_LOW_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/low_final_sequences_ecoli.txt"
PROMODGDE_MID_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/medium_final_sequences_ecoli.txt"
PROMODGDE_HIGH_TXT = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/high_final_sequences_ecoli.txt"

REAL_CSV   = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
TRAIN_CSV  = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp.csv"
MODEL_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"
OUT_DIR    = "/home/yt/Code/DNA-Diffusion/lunwen/tu1_ecoli"

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 2. 工具函数与模型加载
# ══════════════════════════════════════════════════════════════
def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U", "T") if c in "ACGT")

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
    col = next(c for c in ["strength", "expression", "exp"] if c in df.columns)
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(1 / 3)), float(vals.quantile(2 / 3))

def label_fn(x, q1, q2):
    return "low" if x <= q1 else ("mid" if x <= q2 else "high")

def load_real_df(path, q1, q2):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    str_col = next(c for c in ["strength", "expression", "exp"] if c in df.columns)
    seq_col = next(c for c in ["sequence", "seq", "dna"] if c in df.columns)
    df["sequence"] = df[seq_col].apply(standardize)
    df["strength"] = pd.to_numeric(df[str_col], errors="coerce")
    df = df.dropna(subset=["sequence", "strength"])
    df = df[df["sequence"].str.len() > 0].copy()
    df["label"] = df["strength"].apply(lambda x: label_fn(x, q1, q2))
    return df

def load_lstm_predictor():
    exps, seqs = [], []
    with open(TRAIN_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try: exps.append(float(parts[1])); seqs.append(parts[0])
                except ValueError: continue
    scaler = StandardScaler().fit(np.array(exps).reshape(-1, 1))
    seq_len = max(len(s) for s in seqs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=4, hidden_size=256, output_size=1, dropout_rate=0.2, lambda_l2=0.001)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval(), scaler, seq_len, device

def lstm_predict(seqs, model, scaler, seq_len, device, batch_size=256):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            arr = np.zeros((len(batch), seq_len, 4), dtype=np.float32)
            for j, s in enumerate(batch):
                for k, ch in enumerate(s[:seq_len]):
                    if ch in mapping: arr[j, k, mapping[ch]] = 1.0
            preds.append(model(torch.tensor(arr).to(device)).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(preds, axis=0)).flatten()

def condition_acc(scores, target_label, q1, q2, n_boot=500, seed=42):
    hits = np.array([label_fn(s, q1, q2) == target_label for s in scores], dtype=float)
    rng = np.random.RandomState(seed)
    boot = [rng.choice(hits, len(hits), replace=True).mean() for _ in range(n_boot)]
    return hits.mean(), np.std(boot) * 1.96

# ══════════════════════════════════════════════════════════════
# 3. 数据加载 & LSTM 打分逻辑
# ══════════════════════════════════════════════════════════════
print("Step 1 / 3 — Loading sequences")
Q1, Q2 = compute_thresholds(REAL_CSV)
real_df = load_real_df(REAL_CSV, Q1, Q2)

gen_seqs = {
    "our": {"low": read_txt(GEN_LOW_TXT), "mid": read_txt(GEN_MID_TXT), "high": read_txt(GEN_HIGH_TXT)},
    "PromoDGDE": {"low": read_txt(PROMODGDE_LOW_TXT), "mid": read_txt(PROMODGDE_MID_TXT), "high": read_txt(PROMODGDE_HIGH_TXT)},
}

print("\nStep 2 / 3 — LSTM scoring")
model_lstm, scaler_lstm, seq_len_lstm, device_lstm = load_lstm_predictor()

real_scores = {lbl: real_df.loc[real_df["label"] == lbl, "strength"].dropna().values for lbl in LABEL_ORDER}
real_lstm_all = lstm_predict(real_df["sequence"].tolist(), model_lstm, scaler_lstm, seq_len_lstm, device_lstm)
real_df["lstm_pred"] = real_lstm_all

Q1_lstm = float(np.percentile(real_lstm_all, 33.3))
Q2_lstm = float(np.percentile(real_lstm_all, 66.7))

real_lstm_scores = {
    lbl: real_df.loc[real_df["lstm_pred"].apply(lambda x: label_fn(x, Q1_lstm, Q2_lstm)) == lbl, "lstm_pred"].dropna().values
    for lbl in LABEL_ORDER
}

gen_scores = {"our": {}, "PromoDGDE": {}}
for method, seqs_by_lbl in gen_seqs.items():
    for lbl, seqs in seqs_by_lbl.items():
        gen_scores[method][lbl] = lstm_predict(seqs, model_lstm, scaler_lstm, seq_len_lstm, device_lstm)

# 输出条件准确率日志
for method in ["our", "PromoDGDE"]:
    acc = {}
    for lbl in LABEL_ORDER:
        acc[lbl] = condition_acc(gen_scores[method][lbl], lbl, Q1_lstm, Q2_lstm)
    vals = [acc[l][0] for l in LABEL_ORDER]
    overall_mean = float(np.mean(vals))
    overall_ci = float(np.std(vals) / np.sqrt(3) * 1.96)
    print(f"  {method:12s} | low={acc['low'][0]:.3f}±{acc['low'][1]:.3f} | mid={acc['mid'][0]:.3f}±{acc['mid'][1]:.3f} | high={acc['high'][0]:.3f}±{acc['high'][1]:.3f} | overall={overall_mean:.3f}±{overall_ci:.3f}")

# ══════════════════════════════════════════════════════════════
# 4. Fig A — 预测活性分布箱线图
# ══════════════════════════════════════════════════════════════
def plot_fig_a(save_stem="FigA_activity_boxplot_ecoli"):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        GROUP_POS = {"low": 1.0, "mid": 2.0, "high": 3.0}
        OFFSETS   = {"Real": -0.33, "Real_LSTM": -0.11, "PromoDGDE": 0.11, "our": 0.33}
        COLORS    = {"Real": C_REAL, "Real_LSTM": C_REAL_ORC, "PromoDGDE": C_PRO, "our": C_OUR}
        BOX_W     = 0.16

        BP_KW = dict(
            widths=BOX_W,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=1.6, solid_capstyle="round"),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

        all_vals = []
        for lbl in LABEL_ORDER:
            data_map = {
                "Real":      real_scores[lbl],
                "Real_LSTM": real_lstm_scores[lbl],
                "PromoDGDE": gen_scores["PromoDGDE"][lbl],
                "our":       gen_scores["our"][lbl],
            }
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
                ax.scatter(jx, vals[idx], s=3.0, alpha=0.20,
                           color=col, edgecolors="none", zorder=2, rasterized=True)
                all_vals.extend(vals.tolist())

        lo_p = min(all_vals)
        hi_p = np.percentile(all_vals, 99.5)
        pad  = (hi_p - lo_p) * 0.12
        ax.set_ylim(lo_p - pad * 0.5, hi_p + pad)
        ax.set_xticks([GROUP_POS[l] for l in LABEL_ORDER])
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_xlim(0.55, 3.45)
        ax.set_xlabel("Target strength condition")
        ax.set_ylabel("Expression score")

        legend_handles = [
            Patch(facecolor=C_REAL,     edgecolor=C_REAL,     alpha=0.70, linewidth=1.0, linestyle="--", label="Real (measured)"),
            Patch(facecolor=C_REAL_ORC, edgecolor=C_REAL_ORC, alpha=0.70, label="Real (LSTM)"),
            Patch(facecolor=C_PRO,      edgecolor=C_PRO,      alpha=0.70, label="PromoDGDE"),
            Patch(facecolor=C_OUR,      edgecolor=C_OUR,      alpha=0.70, label="ConProDiff"),
        ]
        ax.legend(handles=legend_handles, loc="upper left",
                  handlelength=1.2, handletextpad=0.5, borderpad=0.5, labelspacing=0.30)

        fig.tight_layout(pad=0.5)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 5. Fig B — 混淆矩阵热图
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
            constrained_layout=True,
        )

        for ax, (method_key, method_name) in zip(axes, methods):
            cm = np.zeros((3, 3), dtype=float)
            for i, target_lbl in enumerate(LABEL_ORDER):
                for s in gen_scores[method_key][target_lbl]:
                    pred_lbl = label_fn(s, Q1_lstm, Q2_lstm)
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
                            fontsize=12, color=text_color,
                            fontweight="bold")

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

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 6. 运行主程序
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nStep 3 / 3 — Plotting")
    np.random.seed(42)
    plot_fig_a()
    plot_fig_b()
    print("\nAll figures saved to:", OUT_DIR)