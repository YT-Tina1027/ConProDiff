import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import product
import torch
from sklearn.preprocessing import StandardScaler

PREDICTOR_DIR = "/home/yt/Code/DNA-Diffusion/Predictor"
if PREDICTOR_DIR not in sys.path:
    sys.path.insert(0, PREDICTOR_DIR)
from predictor_models import LSTMModel

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
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
}

LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low Intensity", "mid": "Mid Intensity", "high": "High Intensity"}
COLOR_MAP     = {"low": "#4DBBD5", "mid": "#3C5488", "high": "#E64B35"}

# ══════════════════════════════════════════════════════════════
# 2. 路径配置
# ══════════════════════════════════════════════════════════════
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/high.txt"

REAL_CSV   = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
TRAIN_CSV  = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp.csv"
MODEL_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"
OUT_DIR    = "/home/yt/Code/DNA-Diffusion/lunwen/pearson_ecoli"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 3. 工具函数
# ══════════════════════════════════════════════════════════════
def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U", "T") if c in "ACGT")

def read_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    seqs = []
    for line in open(path):
        tok = max(re.split(r"[\s,\t;|]+", line.strip()), key=lambda x: len(standardize(x)), default="")
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
    str_col = next(c for c in ["strength", "expression", "exp"] if c in df.columns)
    seq_col = next(c for c in ["sequence", "seq", "dna"] if c in df.columns)
    df["sequence"] = df[seq_col].apply(standardize)
    df["strength"] = pd.to_numeric(df[str_col], errors="coerce")
    df = df.dropna(subset=["sequence", "strength"])
    df = df[df["sequence"].str.len() > 0].copy()
    df["label"] = df["strength"].apply(lambda x: label_fn(x, q1, q2))
    return df

def compute_thresholds(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next(c for c in ["strength", "expression", "exp"] if c in df.columns)
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(1/3)), float(vals.quantile(2/3))

def load_lstm_predictor():
    exps, seqs = [], []
    with open(TRAIN_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    exps.append(float(parts[1])); seqs.append(parts[0])
                except ValueError:
                    continue
    scaler  = StandardScaler().fit(np.array(exps).reshape(-1, 1))
    seq_len = max(len(s) for s in seqs)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = LSTMModel(input_size=4, hidden_size=256, output_size=1, dropout_rate=0.2, lambda_l2=0.001)
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
                    if ch in mapping:
                        arr[j, k, mapping[ch]] = 1.0
            preds.append(model(torch.tensor(arr).to(device)).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(preds, axis=0)).flatten()

# ══════════════════════════════════════════════════════════════
# 4. 数据加载
# ══════════════════════════════════════════════════════════════
print("Loading data and scoring with LSTM...")
Q1, Q2 = compute_thresholds(REAL_CSV)
real_df = load_real_df(REAL_CSV, Q1, Q2)

gen_seqs = {
    "low":  read_txt(GEN_LOW_TXT),
    "mid":  read_txt(GEN_MID_TXT),
    "high": read_txt(GEN_HIGH_TXT),
}

model_lstm, scaler_lstm, seq_len_lstm, device_lstm = load_lstm_predictor()

# 真实序列 LSTM 预测分
real_lstm_all = lstm_predict(real_df["sequence"].tolist(), model_lstm, scaler_lstm, seq_len_lstm, device_lstm)
real_df["lstm_pred"] = real_lstm_all
Q1_lstm = float(np.percentile(real_lstm_all, 33.3))
Q2_lstm = float(np.percentile(real_lstm_all, 66.7))

# 生成序列 LSTM 预测分
gen_scores = {
    lbl: lstm_predict(seqs, model_lstm, scaler_lstm, seq_len_lstm, device_lstm)
    for lbl, seqs in gen_seqs.items()
}

# 真实序列按 LSTM 预测分重新分组
real_lstm_scores = {
    lbl: real_df.loc[
        real_df["lstm_pred"].apply(lambda x: label_fn(x, Q1_lstm, Q2_lstm)) == lbl,
        "lstm_pred"
    ].values
    for lbl in LABEL_ORDER
}

# ══════════════════════════════════════════════════════════════
# 5. Fig A 数据构建
#    逻辑：同一强度组内，对真实序列和生成序列各取相同数量，
#    逐条配对后绘制 real_lstm vs gen_lstm 散点图。
# ══════════════════════════════════════════════════════════════
def build_intensity_df():
    rows = []
    for lbl in LABEL_ORDER:
        real_vals = real_lstm_scores[lbl]
        gen_vals  = gen_scores[lbl]
        # 取较小长度，随机采样配对
        n = min(len(real_vals), len(gen_vals))
        rng = np.random.default_rng(42)
        r_idx = rng.choice(len(real_vals), n, replace=False)
        g_idx = rng.choice(len(gen_vals),  n, replace=False)
        for r, g in zip(real_vals[r_idx], gen_vals[g_idx]):
            rows.append({"strength_label": lbl, "real_intensity": r, "gen_intensity": g})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# 6. Fig B 数据构建：真实序列 & 生成序列各组 k-mer 频率
# ══════════════════════════════════════════════════════════════
def compute_kmer_freq(seqs, k):
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    counts = {km: 0 for km in kmers}
    total = 0
    for s in seqs:
        for i in range(len(s) - k + 1):
            sub = s[i:i+k]
            if sub in counts:
                counts[sub] += 1
                total += 1
    freq = np.array([counts[km] / total if total > 0 else 0.0 for km in kmers])
    return kmers, freq

def build_kmer_df(kmer_lengths=[3, 4, 5]):
    rows = []
    for lbl in LABEL_ORDER:
        real_seqs_lbl = real_df.loc[real_df["label"] == lbl, "sequence"].tolist()
        gen_seqs_lbl  = gen_seqs[lbl]
        for k in kmer_lengths:
            kmers, real_freq = compute_kmer_freq(real_seqs_lbl, k)
            _,     gen_freq  = compute_kmer_freq(gen_seqs_lbl,  k)
            for km, rf, gf in zip(kmers, real_freq, gen_freq):
                rows.append({
                    "strength_label": lbl, "k": k, "kmer": km,
                    "real_freq": rf, "gen_freq": gf,
                })
    return pd.concat([pd.DataFrame([r]) for r in rows], ignore_index=True)

# ══════════════════════════════════════════════════════════════
# 5. Fig A 数据构建（不再配对，各自独立保存）
# ══════════════════════════════════════════════════════════════
def plot_intensity_distribution(
    real_scores_dict, gen_scores_dict,
    save_stem="FigA_intensity_distribution"
):
    from scipy.stats import gaussian_kde, ks_2samp

    FIG_W, FIG_H = 6.6, 2.50
    ALPHA_REAL = 0.25
    ALPHA_GEN  = 0.10

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))

        for ax, lbl in zip(axes, LABEL_ORDER):
            color  = COLOR_MAP[lbl]
            r_vals = real_scores_dict[lbl]
            g_vals = gen_scores_dict[lbl]

            all_vals = np.concatenate([r_vals, g_vals])
            x_min, x_max = all_vals.min(), all_vals.max()
            margin = (x_max - x_min) * 0.10
            xs = np.linspace(x_min - margin, x_max + margin, 500)

            kde_r = gaussian_kde(r_vals, bw_method="scott")
            kde_g = gaussian_kde(g_vals, bw_method="scott")
            yr = kde_r(xs)
            yg = kde_g(xs)

            # OVL 用面积归一化版（概率密度，面积=1）计算
            overlap = float(np.trapz(np.minimum(yr, yg), xs))

            # 峰高归一化用于画图
            peak = max(yr.max(), yg.max())
            yr_n = yr / peak
            yg_n = yg / peak

            ax.fill_between(xs, yr_n, alpha=ALPHA_REAL, color=color, zorder=2)
            ax.plot(xs, yr_n, lw=1.2, color=color,linestyle="--", dashes=(4, 3),
                    label="Real", zorder=3)

            ax.fill_between(xs, yg_n, alpha=ALPHA_GEN, color=color, zorder=1)
            ax.plot(xs, yg_n, lw=1.2, color=color,
                label="Generated", zorder=3)

            ax.axvline(np.mean(r_vals), color=color,
                       lw=0.8, linestyle="--", dashes=(4, 4), alpha=0.55, zorder=4)
            ax.axvline(np.mean(g_vals), color=color,
                       lw=0.8, linestyle="-",  alpha=0.55, zorder=4)

            ks_stat, ks_p = ks_2samp(r_vals, g_vals)
            p_str = "p < 0.001" if ks_p < 0.001 else f"p = {ks_p:.3f}"
            ax.text(0.97, 0.96,
                    f"KS = {ks_stat:.3f}\n{p_str}\nOVL = {overlap:.3f}",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=6.0,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="none", alpha=0.8))

            ax.fill_between(xs, np.minimum(yr_n, yg_n),
                            alpha=0.30, color="#888888",
                            label="Overlap", zorder=2)

            ax.legend(loc="upper left", fontsize=5.5,
                      handlelength=2.2, handletextpad=0.3,
                      borderpad=0.3, labelspacing=0.3,
                      frameon=False)

            ax.set_title(LABEL_DISPLAY[lbl], fontsize=7.5, pad=4)
            ax.set_ylim(-0.02, 1.12)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.ticklabel_format(style="sci", axis="x",
                                scilimits=(0, 0), useOffset=False)
            ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
            ax.set_box_aspect(1)   # ← 正方形

        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.97,
                            top=0.88, wspace=0.45)
        fig.supxlabel("LSTM Predicted Score", fontsize=8, y=0.01)
        fig.supylabel("Density (normalized)",  fontsize=8, x=0.01)

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f" -> 已保存: {save_stem}.{ext}")
        plt.close(fig)


def _draw_scatter(ax, x, y, color, rasterized=False):
    r, p = stats.pearsonr(x, y)
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx],
            color="#999999", linestyle="--", linewidth=0.6, zorder=1)
    ax.scatter(x, y, s=4, alpha=0.48, color=color,
               edgecolors="none", zorder=2, rasterized=rasterized)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(
        0.05, 0.95,
        f"$R^2={r**2:.3f}$\n$r={r:.3f}$\n{p_str}",
        transform=ax.transAxes, va="top", fontsize=7.0,   # ← 6.0→7.0
        bbox=dict(boxstyle="round,pad=0.15",
                  facecolor="white", edgecolor="none", alpha=0.7),
    )
    ax.set_title("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    return r, p


def plot_kmer_correlation(kmer_df, kmer_lengths=[3, 4, 5], save_stem="FigB_kmer_correlation"):
    FIG_W, FIG_H = 7.20, 7.20   # ← 高度从6.60→7.20，配合正方形
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(3, 3, figsize=(FIG_W, FIG_H))
        for row_idx, lbl in enumerate(LABEL_ORDER):
            for col_idx, k in enumerate(kmer_lengths):
                ax  = axes[row_idx, col_idx]
                sub = kmer_df[(kmer_df["strength_label"] == lbl) & (kmer_df["k"] == k)]
                if sub.empty:
                    continue
                _draw_scatter(ax, sub["gen_freq"].values, sub["real_freq"].values,
                              COLOR_MAP[lbl], rasterized=(k >= 5))
                ax.set_aspect(1.0, adjustable="box")   # ← 正方形
                ax.ticklabel_format(style='sci', axis='both',
                                    scilimits=(-2, -2), useOffset=False)
                ax.xaxis.get_offset_text().set_fontsize(6.5)   # ← 5.0→6.5
                ax.yaxis.get_offset_text().set_fontsize(6.5)
                ax.set_xlabel("")
                ax.set_ylabel("")
                if row_idx == 2:
                    ax.set_xlabel(f"{k}-mer", labelpad=3, fontsize=8.5)   # ← 7.5→8.5
                if col_idx == 0:
                    ax.set_ylabel(LABEL_DISPLAY[lbl], labelpad=4, fontsize=8.5)
        fig.subplots_adjust(left=0.11, bottom=0.09, right=0.97,
                            top=0.97, hspace=0.38, wspace=0.38)
        fig.supxlabel("k-mer Frequency (Generated)", fontsize=9, y=0.01)
        fig.supylabel("k-mer Frequency (Real)",      fontsize=9, x=0.01)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f" -> 已保存: {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 10. 主程序
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Fig A：分布对比（不再需要配对 DataFrame）
    print("渲染 Fig A（分布对比）...")
    plot_intensity_distribution(real_lstm_scores, gen_scores)

    print("构建 Fig B k-mer 数据...")
    kmer_df = build_kmer_df(kmer_lengths=[3, 4, 5])

    print("渲染 Fig B...")
    plot_kmer_correlation(kmer_df)

    print("全部完成。")