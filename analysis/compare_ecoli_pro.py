# -*- coding: utf-8 -*-
import os, sys, re, warnings, logging
import gc as _gc
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════
# 0.  Nature-style matplotlib theme
# ══════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

PAL = {
    "our":       "#E64B35",   # Vermilion
    "PromoDGDE": "#4DBBD5",   # Teal
    "Real":      "#3C5488",   # Navy
}

LABEL_ORDER  = ["low", "mid", "high"]
METHOD_ORDER = ["Real", "our", "PromoDGDE"]
GEN_METHODS  = ["our", "PromoDGDE"]
METHOD_LABEL = {"our": "Our model", "PromoDGDE": "PromoDGDE"}

np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════
# 1.  Paths
# ══════════════════════════════════════════════════════════════

GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/exiperiments/ablation_full/data_full/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/exiperiments/ablation_full/data_full/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/exiperiments/ablation_full/data_full/high.txt"

PROMODGDE_LOW_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/low_final_sequences_ecoli.txt"
PROMODGDE_MID_TXT  = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/medium_final_sequences_ecoli.txt"
PROMODGDE_HIGH_TXT = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/high_final_sequences_ecoli.txt"

REAL_CSV      = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
TRAIN_CSV     = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp.csv"
PREDICTOR_DIR = "/home/yt/Code/DNA-Diffusion/Predictor"
MODEL_PATH    = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"

OUT_DIR    = "/home/yt/Code/DNA-Diffusion/compare_ecoli_pub1"
N_PER_BIN  = 1000

os.makedirs(OUT_DIR, exist_ok=True)

if PREDICTOR_DIR not in sys.path:
    sys.path.insert(0, PREDICTOR_DIR)
from predictor_models import LSTMModel

# ══════════════════════════════════════════════════════════════
# 2.  Helpers
# ══════════════════════════════════════════════════════════════

def standardize(s):
    return "".join(c for c in str(s).upper().strip().replace("U", "T") if c in "ACGT")

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

def gc(seq):
    seq = standardize(seq)
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else np.nan

def all_kmers(k):
    return ["".join(p) for p in product("ACGT", repeat=k)]

def kmer_freq(seqs, k):
    vocab = all_kmers(k)
    cnt, tot = Counter(), 0
    for s in seqs:
        s = standardize(s)
        for i in range(len(s) - k + 1):
            w = s[i:i + k]
            if set(w) <= set("ACGT"):
                cnt[w] += 1; tot += 1
    if tot == 0:
        return np.zeros(len(vocab))
    return np.array([cnt[v] / tot for v in vocab])

def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def mwu(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p

def stars(p):
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"

def success_rate(scores, lo, hi):
    scores = np.asarray(scores)
    return float(np.mean((scores > lo) & (scores <= hi)))

def savefig(fig, stem):
    fig.savefig(os.path.join(OUT_DIR, stem + ".png"))
    fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"))
    plt.close(fig)
    print(f"  saved {stem}.png / .pdf")

def mean_edit_dist(seqs, n_sample=200, seed=42):
    import Levenshtein
    pool = list(seqs)
    if len(pool) > n_sample:
        rs = np.random.RandomState(seed)
        idx = rs.choice(len(pool), n_sample, replace=False)
        pool = [pool[i] for i in idx]
    if len(pool) < 2:
        return np.nan
    dists = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            dists.append(Levenshtein.distance(pool[i], pool[j]))
    return float(np.mean(dists))

# ══════════════════════════════════════════════════════════════
# 3.  Data loading
# ══════════════════════════════════════════════════════════════

def compute_thresholds(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next((c for c in ["strength", "expression", "exp"] if c in df.columns), None)
    if col is None:
        raise ValueError(f"No strength column in {path}")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(1 / 3)), float(vals.quantile(2 / 3))

def label_fn(x, q1, q2):
    if x <= q1: return "low"
    if x <= q2: return "mid"
    return "high"

def load_real(path, q1, q2):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    str_col = next((c for c in ["strength", "expression", "exp"] if c in df.columns), None)
    seq_col = next((c for c in ["sequence", "seq", "dna"] if c in df.columns), None)
    df["sequence"] = df[seq_col].apply(standardize)
    df["strength"] = pd.to_numeric(df[str_col], errors="coerce")
    df = df.dropna(subset=["sequence", "strength"])
    df = df[df["sequence"].str.len() > 0]
    df["label"]        = df["strength"].apply(lambda x: label_fn(x, q1, q2))
    df["method"]       = "Real"
    df["oracle_score"] = df["strength"]
    return df[["method", "label", "sequence", "strength", "oracle_score"]].copy()

def load_generated():
    rows = []
    for method, label, path in [
        ("our",       "low",  GEN_LOW_TXT),
        ("our",       "mid",  GEN_MID_TXT),
        ("our",       "high", GEN_HIGH_TXT),
        ("PromoDGDE", "low",  PROMODGDE_LOW_TXT),
        ("PromoDGDE", "mid",  PROMODGDE_MID_TXT),
        ("PromoDGDE", "high", PROMODGDE_HIGH_TXT),
    ]:
        for s in read_txt(path):
            rows.append({"method": method, "label": label, "sequence": s,
                         "strength": np.nan, "oracle_score": np.nan})
    return pd.DataFrame(rows)

def downsample(df, n=1000, seed=42):
    parts = []
    for (m, l), sub in df.groupby(["method", "label"]):
        parts.append(sub.sample(min(len(sub), n), random_state=seed))
    return pd.concat(parts, ignore_index=True)

# ══════════════════════════════════════════════════════════════
# 4.  LSTM predictor (replaces TF oracle)
# ══════════════════════════════════════════════════════════════

def load_lstm_predictor():
    seqs, exps = [], []
    with open(TRAIN_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    exps.append(float(parts[1])); seqs.append(parts[0])
                except ValueError:
                    continue
    scaler = StandardScaler()
    scaler.fit(np.array(exps).reshape(-1, 1))
    seq_len = max(len(s) for s in seqs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=4, hidden_size=256, output_size=1,
                      dropout_rate=0.2, lambda_l2=0.001)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, scaler, seq_len, device

def lstm_predict(seqs, model, scaler, seq_len, device, batch_size=256):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            arr = np.zeros((len(batch), seq_len, 4), dtype=np.float32)
            for j, s in enumerate(batch):
                for k_idx, ch in enumerate(s[:seq_len]):
                    if ch in mapping:
                        arr[j, k_idx, mapping[ch]] = 1.0
            t = torch.tensor(arr).to(device)
            preds.append(model(t).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return scaler.inverse_transform(preds).flatten()

def attach_predictor_scores(df):
    print("Loading E. coli LSTM predictor …")
    model, scaler, seq_len, device = load_lstm_predictor()
    parts = []
    for (method, label), sub in df.groupby(["method", "label"]):
        sub = sub.copy()
        if method == "Real":
            parts.append(sub)
            continue
        seqs = sub["sequence"].tolist()
        print(f"  scoring {method} | {label} | n={len(seqs)}")
        sub["oracle_score"] = lstm_predict(seqs, model, scaler, seq_len, device)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)

# ══════════════════════════════════════════════════════════════
# 5.  Metric builders
# ══════════════════════════════════════════════════════════════

def build_metrics(df, TR):
    rows = []
    for method in GEN_METHODS:
        for label in LABEL_ORDER:
            sub  = df[(df.method == method) & (df.label == label)]
            real = df[(df.method == "Real")  & (df.label == label)]
            sc   = sub["oracle_score"].dropna().values
            rc   = real["oracle_score"].dropna().values
            lo, hi = TR[label]
            p_val = mwu(sc, rc)
            rows.append({
                "Method":         METHOD_LABEL[method],
                "Condition":      label,
                "n":              len(sc),
                "Oracle mean±SD": f"{np.mean(sc):.3f}±{np.std(sc):.3f}",
                "Oracle median":  round(float(np.median(sc)), 3),
                "Success rate":   round(success_rate(sc, lo, hi), 3),
                "GC mean":        round(float(np.nanmean([gc(s) for s in sub.sequence])), 3),
                "MWU vs Real":    stars(p_val),
            })
    return pd.DataFrame(rows)

def build_kmer_table(df):
    rows = []
    for k in [3, 4, 5]:
        for label in LABEL_ORDER:
            real_seqs = df[(df.method == "Real")  & (df.label == label)].sequence.tolist()
            rv = kmer_freq(real_seqs, k)
            for method in GEN_METHODS:
                gen_seqs = df[(df.method == method) & (df.label == label)].sequence.tolist()
                gv = kmer_freq(gen_seqs, k)
                rows.append({
                    "k": k, "Condition": label,
                    "Method": METHOD_LABEL[method],
                    "Pearson r": round(pearson(rv, gv), 4),
                })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# 6.  LaTeX table helper
# ══════════════════════════════════════════════════════════════

def to_latex(df, path, caption="", label=""):
    col_fmt = "l" * len(df.columns)
    lines = [
        r"\begin{table}[ht]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\small", r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        " & ".join(str(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  saved {os.path.basename(path)}")

# ══════════════════════════════════════════════════════════════
# 7.  Fig 1 – Oracle violin (3 violins per condition)
# ══════════════════════════════════════════════════════════════

def fig_violin(df, TR):
    ALL_METHODS = ["Real"] + GEN_METHODS
    ALL_COLORS  = {"Real": "#888888", "our": PAL["our"], "PromoDGDE": PAL["PromoDGDE"]}
    ALL_ALPHA   = {"Real": 0.22, "our": 0.38, "PromoDGDE": 0.38}

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.8), sharey=False)

    for ax, label in zip(axes, LABEL_ORDER):
        lo, hi = TR[label]

        for pos, method in enumerate(ALL_METHODS, start=1):
            vals = df[(df.method == method) & (df.label == label)]["oracle_score"].dropna().values
            if len(vals) == 0:
                continue
            col = ALL_COLORS[method]

            vp = ax.violinplot(vals, positions=[pos], widths=0.60,
                               showmedians=False, showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(col); pc.set_alpha(ALL_ALPHA[method])
                pc.set_edgecolor(col); pc.set_linewidth(0.9)

            q1v, med, q3v = np.percentile(vals, [25, 50, 75])
            ax.plot([pos, pos], [q1v, q3v], color=col,
                    lw=2.6, solid_capstyle="round", zorder=4)
            ax.plot(pos, med, "o", color=col, ms=4.5, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.5)

            if method != "Real":
                n_strip = min(len(vals), 250)
                sample  = vals if len(vals) <= 250 else np.random.choice(vals, 250, replace=False)
                jitter  = np.random.uniform(-0.16, 0.16, size=n_strip)
                ax.scatter(pos + jitter, sample, s=2.2,
                           color=col, alpha=0.20, lw=0, zorder=2)

        real_med = float(df[(df.method == "Real") & (df.label == label)]["oracle_score"].median())
        ax.axhline(real_med, color="#555555", lw=1.1, ls="--", alpha=0.75, zorder=3)

        ax.autoscale(axis="y", tight=False)
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.06
        ymin -= pad; ymax += pad
        ax.set_ylim(ymin, ymax)

        shade_lo = max(lo if np.isfinite(lo) else ymin, ymin)
        shade_hi = min(hi if np.isfinite(hi) else ymax, ymax)
        ax.axhspan(shade_lo, shade_hi, color="#FFD700", alpha=0.13, zorder=0)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Real", METHOD_LABEL["our"], METHOD_LABEL["PromoDGDE"]],
                           rotation=20, ha="right", fontsize=7.5)
        ax.set_title(label.capitalize(), fontweight="bold", pad=5)
        ax.set_xlim(0.4, 3.6)

    axes[0].set_ylabel("LSTM expression score")

    handles = [
        mpatches.Patch(facecolor=ALL_COLORS[m], alpha=0.55,
                       label="Real data" if m == "Real" else METHOD_LABEL[m])
        for m in ALL_METHODS
    ] + [
        plt.Line2D([0], [0], color="#555555", ls="--", lw=1.2, label="Real median"),
        mpatches.Patch(facecolor="#FFD700", alpha=0.35, label="Target zone"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.07), frameon=False, fontsize=8)
    fig.suptitle("LSTM expression score distribution by target condition (E. coli)",
                 y=1.12, fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "Fig1_oracle_violin")

# ══════════════════════════════════════════════════════════════
# 8.  Fig 2 – Success rate + Intra-group edit distance
# ══════════════════════════════════════════════════════════════

def fig_success_and_diversity(df, TR):
    """
    左图：target success rate（条件可控性）
    右图：intra-group edit distance（序列多样性）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    x     = np.arange(len(LABEL_ORDER))
    width = 0.32

    # ── 左图：Success rate ────────────────────────────────────
    for i, method in enumerate(GEN_METHODS):
        vals, errs = [], []
        for label in LABEL_ORDER:
            lo, hi = TR[label]
            sc = df[(df.method == method) & (df.label == label)]["oracle_score"].dropna().values
            if len(sc) == 0:
                vals.append(np.nan); errs.append(0); continue
            sr   = success_rate(sc, lo, hi)
            boot = [success_rate(np.random.choice(sc, len(sc), replace=True), lo, hi)
                    for _ in range(500)]
            vals.append(sr)
            errs.append(np.std(boot) * 1.96)

        xs   = x + (i - 0.5) * width
        bars = ax1.bar(xs, vals, width=width,
                       color=PAL[method], alpha=0.85,
                       edgecolor="white", linewidth=0.4,
                       label=METHOD_LABEL[method], zorder=3)
        ax1.errorbar(xs, vals, yerr=errs,
                     fmt="none", color="black", capsize=3, lw=0.8, zorder=4)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                         f"{val:.2f}", ha="center", va="bottom",
                         fontsize=7, fontweight="bold")

    # significance between methods
    for i, label in enumerate(LABEL_ORDER):
        lo, hi = TR[label]
        sc = {m: df[(df.method == m) & (df.label == label)]["oracle_score"].dropna().values
              for m in GEN_METHODS}
        p  = mwu(sc[GEN_METHODS[0]], sc[GEN_METHODS[1]])
        st = stars(p)
        if st not in ("ns", ""):
            top = max(success_rate(sc[GEN_METHODS[0]], lo, hi),
                      success_rate(sc[GEN_METHODS[1]], lo, hi))
            ax1.text(i, top + 0.10, st, ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([l.capitalize() for l in LABEL_ORDER])
    ax1.set_ylim(0, 1.18)
    ax1.set_xlabel("Target condition")
    ax1.set_ylabel("Target success rate")
    ax1.set_title("Conditional controllability",pad=14, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.legend(loc="upper right", fontsize=7.5)

    # ── 右图：Edit distance ───────────────────────────────────
    real_eds = []
    for label in LABEL_ORDER:
        seqs = df[(df.method == "Real") & (df.label == label)].sequence.tolist()
        real_eds.append(mean_edit_dist(seqs))

    for i, method in enumerate(GEN_METHODS):
        vals, errs = [], []
        for label in LABEL_ORDER:
            seqs = df[(df.method == method) & (df.label == label)].sequence.tolist()
            ed = mean_edit_dist(seqs)
            rs_boot = np.random.RandomState(42)
            boots = []
            pool = seqs[:200]
            for _ in range(200):
                s = rs_boot.choice(len(pool), len(pool), replace=True)
                sample = [pool[k] for k in s]
                boots.append(mean_edit_dist(sample, n_sample=80))
            vals.append(ed)
            errs.append(np.std(boots) * 1.96)

        xs   = x + (i - 0.5) * width
        bars = ax2.bar(xs, vals, width=width,
                       color=PAL[method], alpha=0.85,
                       edgecolor="white", linewidth=0.4,
                       label=METHOD_LABEL[method], zorder=3)
        ax2.errorbar(xs, vals, yerr=errs,
                     fmt="none", color="black", capsize=3, lw=0.8, zorder=4)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         val + max(errs) * 0.15 + 0.3,
                         f"{val:.1f}", ha="center", va="bottom",
                         fontsize=7, fontweight="bold")

    for ci, (label, red) in enumerate(zip(LABEL_ORDER, real_eds)):
        if not np.isnan(red):
            ax2.plot([ci - 0.38, ci + 0.38], [red, red],
                     color="#3C5488", lw=1.6, ls="--", zorder=5)

    # significance
    for i, label in enumerate(LABEL_ORDER):
        eds = {}
        for method in GEN_METHODS:
            seqs = df[(df.method == method) & (df.label == label)].sequence.tolist()
            pool = seqs[:200]
            rs2  = np.random.RandomState(0)
            sub  = pool if len(pool) <= 50 else [pool[k] for k in rs2.choice(len(pool), 50, replace=False)]
            row_means = []
            for ii, a in enumerate(sub):
                ds = [mean_edit_dist([a, sub[jj]], n_sample=2)
                      for jj in range(len(sub)) if jj != ii]
                row_means.append(np.mean(ds) if ds else np.nan)
            eds[method] = np.array(row_means)
        p  = mwu(eds[GEN_METHODS[0]], eds[GEN_METHODS[1]])
        st = stars(p)
        if st not in ("ns", ""):
            top = max(
                mean_edit_dist(df[(df.method == GEN_METHODS[0]) & (df.label == label)].sequence.tolist()),
                mean_edit_dist(df[(df.method == GEN_METHODS[1]) & (df.label == label)].sequence.tolist()),
            )
            ax2.text(i, top + 2.5, st, ha="center", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([l.capitalize() for l in LABEL_ORDER])
    ax2.set_xlabel("Target condition")
    ax2.set_ylabel("Mean intra-group edit distance")
    ax2.set_title("Sequence diversity (edit distance)", pad=14, fontweight="bold")

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2.append(plt.Line2D([0], [0], color="#3C5488", ls="--", lw=1.6))
    labels2.append("Real (baseline)")
    ax2.legend(handles2, labels2, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.07), fontsize=7.5)

    fig.tight_layout()
    savefig(fig, "Fig2_controllability_diversity")

# ══════════════════════════════════════════════════════════════
# 9.  Fig 3 – GC content box + 4-mer bar panel
# ══════════════════════════════════════════════════════════════

def fig_gc_kmer(df):
    fig = plt.figure(figsize=(9.0, 3.8))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # GC boxplot
    df2 = df.copy()
    df2["gc_val"] = df2["sequence"].apply(gc)
    base = {"low": 1.0, "mid": 2.0, "high": 3.0}
    off  = {"Real": -0.25, "our": 0.0, "PromoDGDE": 0.25}

    for label in LABEL_ORDER:
        for method in METHOD_ORDER:
            vals = df2[(df2.method == method) & (df2.label == label)]["gc_val"].dropna().values
            if len(vals) == 0: continue
            pos = base[label] + off[method]
            col = PAL.get(method, "#888888")
            ax1.boxplot(vals, positions=[pos], widths=0.18,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=col, alpha=0.35, edgecolor=col, lw=1.0),
                        medianprops=dict(color=col, lw=1.5),
                        whiskerprops=dict(color=col, lw=0.8),
                        capprops=dict(color=col, lw=0.8))

    ax1.set_xlim(0.5, 3.5)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels([l.capitalize() for l in LABEL_ORDER])
    ax1.set_xlabel("Target condition")
    ax1.set_ylabel("GC content")
    ax1.set_title("GC content distribution", pad=14, fontweight="bold")
    handles_gc = [mpatches.Patch(facecolor=PAL.get(m, "#888"), alpha=0.5,
                                  label=METHOD_LABEL.get(m, m))
                  for m in METHOD_ORDER]
    ax1.legend(handles=handles_gc, loc="upper right", fontsize=7)

    # 4-mer bar
    k = 4
    x, width = np.arange(len(LABEL_ORDER)), 0.32
    for i, method in enumerate(GEN_METHODS):
        vals = []
        for label in LABEL_ORDER:
            rv = kmer_freq(df[(df.method == "Real")   & (df.label == label)].sequence.tolist(), k)
            gv = kmer_freq(df[(df.method == method) & (df.label == label)].sequence.tolist(), k)
            vals.append(pearson(rv, gv))
        xs   = x + (i - 0.5) * width
        bars = ax2.bar(xs, vals, width=width, color=PAL[method], alpha=0.85,
                       edgecolor="white", lw=0.4, label=METHOD_LABEL[method])
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([l.capitalize() for l in LABEL_ORDER])
    ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("Target condition")
    ax2.set_ylabel("Pearson r with real 4-mer distribution")
    ax2.set_title("4-mer distributional fidelity", pad=14, fontweight="bold")
    ax2.legend(loc="lower right", ncol=2,  fontsize=7,bbox_to_anchor=(1.00, 0.97))

    fig.tight_layout()
    savefig(fig, "Fig3_gc_kmer_panel")

# ══════════════════════════════════════════════════════════════
# 10.  Fig 4 – Three-column metric heatmap with Δ column
#       Rows: Success rate | Edit dist | 3-mer r | 4-mer r | 5-mer r | GC diff
# ══════════════════════════════════════════════════════════════

def fig_heatmap(df, TR):
    METRICS = ["Success\nrate", "Edit\ndist", "3-mer r", "4-mer r", "5-mer r", "GC\ndiff"]
    HIB     = [True, True, True, True, True, False]   # higher-is-better
    FMT     = ["%", "f1", "f3", "f3", "f3", "f3"]
    N_M, N_C = len(METRICS), len(LABEL_ORDER)

    raw = {m: np.full((N_M, N_C), np.nan) for m in GEN_METHODS}

    for ci, label in enumerate(LABEL_ORDER):
        lo, hi    = TR[label]
        real_seqs = df[(df.method == "Real") & (df.label == label)].sequence.tolist()
        real_gc_v = float(np.nanmean([gc(s) for s in real_seqs])) if real_seqs else np.nan
        rv3 = kmer_freq(real_seqs, 3)
        rv4 = kmer_freq(real_seqs, 4)
        rv5 = kmer_freq(real_seqs, 5)

        for method in GEN_METHODS:
            sub  = df[(df.method == method) & (df.label == label)]
            sc   = sub["oracle_score"].dropna().values
            seqs = sub.sequence.tolist()
            sr   = success_rate(sc, lo, hi) if len(sc) else np.nan
            ed   = mean_edit_dist(seqs)     if seqs    else np.nan
            k3r  = pearson(rv3, kmer_freq(seqs, 3))
            k4r  = pearson(rv4, kmer_freq(seqs, 4))
            k5r  = pearson(rv5, kmer_freq(seqs, 5))
            gcd  = abs(float(np.nanmean([gc(s) for s in seqs])) - real_gc_v) if seqs else np.nan
            raw[method][:, ci] = [sr, ed, k3r, k4r, k5r, gcd]

    # Δ matrix (positive = Our wins)
    delta = np.full((N_M, N_C), np.nan)
    for ri in range(N_M):
        for ci in range(N_C):
            a, b = raw["our"][ri, ci], raw["PromoDGDE"][ri, ci]
            if not (np.isnan(a) or np.isnan(b)):
                delta[ri, ci] = (a - b) if HIB[ri] else (b - a)

    # Row-normalise: shared scale between two methods
    def row_norm(mat_list):
        n_rows    = mat_list[0].shape[0]
        norm_mats = [np.full(m.shape, 0.5, dtype=np.float64) for m in mat_list]
        for ri in range(n_rows):
            all_v = np.concatenate([m[ri, :] for m in mat_list])
            all_v = all_v[~np.isnan(all_v)]
            if len(all_v) == 0:
                continue
            if ri == 1:
                # Edit dist 行：用绝对范围避免微小差异被放大
                center = float(np.nanmean(all_v))
                vlo, vhi = center * 0.85, center * 1.15
            else:
                vlo, vhi = all_v.min(), all_v.max()
            rng = vhi - vlo
            for mat, norm_mat in zip(mat_list, norm_mats):
                if rng > 1e-9:
                    norm_mat[ri, :] = np.clip((mat[ri, :] - vlo) / rng, 0, 1)
                else:
                    norm_mat[ri, :] = 0.75
                if not HIB[ri]:
                    norm_mat[ri, :] = 1.0 - norm_mat[ri, :]
        return norm_mats

    norm_our, norm_pro = row_norm([raw["our"], raw["PromoDGDE"]])

    # Format cell text
    def fmt_val(v, ri):
        if np.isnan(v): return "–"
        if FMT[ri] == "%":  return f"{v:.1%}"
        if FMT[ri] == "f1": return f"{v:.1f}"
        return f"{v:.3f}"

    def fmt_delta(v, ri):
        if np.isnan(v): return "–"
        sign = "+" if v > 0 else ""
        if FMT[ri] == "%":  return f"{sign}{v:.1%}"
        if FMT[ri] == "f1": return f"{sign}{v:.1f}"
        return f"{sign}{v:.3f}"

    fig = plt.figure(figsize=(9.6, 5.2))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.06, width_ratios=[2, 2, 1.4])
    ax_our   = fig.add_subplot(gs[0])
    ax_pro   = fig.add_subplot(gs[1])
    ax_delta = fig.add_subplot(gs[2])

    def draw_method_heatmap(ax, mat, norm_mat, method, show_yticks):
        light = "#FADADD" if method == "our" else "#D0EBF5"
        cmap  = mcolors.LinearSegmentedColormap.from_list(
            method, [light, PAL[method]], N=256)
        ax.imshow(norm_mat, cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")

        for ri in range(N_M):
            for ci in range(N_C):
                v   = mat[ri, ci]
                txt = fmt_val(v, ri)
                brightness = norm_mat[ri, ci]
                fc = "white" if brightness > 0.55 else "#1a1a1a"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=9, color=fc, fontweight="bold")

        ax.set_xticks(range(N_C))
        ax.set_xticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=9)
        ax.set_xlabel("Target condition", fontsize=9)
        ax.set_title(METHOD_LABEL[method], fontweight="bold",
                     fontsize=10, pad=6, color=PAL[method])

        if show_yticks:
            ax.set_yticks(range(N_M))
            ax.set_yticklabels(METRICS, fontsize=9)
        else:
            ax.set_yticks([])

        for x in np.arange(-0.5, N_C, 1):
            ax.axvline(x, color="white", lw=1.5)
        for y in np.arange(-0.5, N_M, 1):
            ax.axhline(y, color="white", lw=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#cccccc")

    draw_method_heatmap(ax_our, raw["our"],       norm_our, "our",       show_yticks=True)
    draw_method_heatmap(ax_pro, raw["PromoDGDE"], norm_pro, "PromoDGDE", show_yticks=False)

    # Δ column
    cmap_div = mcolors.LinearSegmentedColormap.from_list(
        "delta", [PAL["PromoDGDE"], "#f7f7f7", PAL["our"]], N=256)
    delta_norm = np.full_like(delta, 0.5, dtype=np.float64)
    for ri in range(N_M):
        row = delta[ri, :]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        ram = max(np.nanmax(np.abs(valid)), 1e-9)
        delta_norm[ri, :] = np.where(
            np.isnan(row),
            np.nan,
            np.clip((row + ram) / (2 * ram), 0, 1)
        )

    ax_delta.imshow(delta_norm, cmap=cmap_div,
                    vmin=0, vmax=1,
                    aspect="auto", interpolation="nearest")

    for ri in range(N_M):
        for ci in range(N_C):
            v = delta[ri, ci]
            txt = fmt_delta(v, ri)
            if np.isnan(v):
                fc = "#555555"
            else:
                norm_v = delta_norm[ri, ci]
                fc = "white" if (norm_v > 0.72 or norm_v < 0.28) else "#1a1a1a"
            ax_delta.text(ci, ri, txt, ha="center", va="center",
                          fontsize=8.5, color=fc, fontweight="bold")

    ax_delta.set_xticks(range(N_C))
    ax_delta.set_xticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=9)
    ax_delta.set_xlabel("Target condition", fontsize=9)
    ax_delta.set_yticks([])
    ax_delta.set_title("Δ (Ours − PromoDGDE)\n", fontweight="bold",
                       fontsize=9, pad=2, color="#444444")

    for x in np.arange(-0.5, N_C, 1):
        ax_delta.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, N_M, 1):
        ax_delta.axhline(y, color="white", lw=1.5)
    for spine in ax_delta.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#cccccc")

    leg_handles = [
        mpatches.Patch(facecolor=PAL["our"],       label="Our model wins  (+)"),
        mpatches.Patch(facecolor=PAL["PromoDGDE"], label="PromoDGDE wins  (−)"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.72, -0.04), frameon=False, fontsize=8)

    fig.suptitle("Multi-metric comparison: Our model vs. PromoDGDE (E. coli)",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "Fig4_metric_heatmap")

# ══════════════════════════════════════════════════════════════
# 11.  Tables
# ══════════════════════════════════════════════════════════════

def make_tables(df, TR):
    metrics = build_metrics(df, TR)
    metrics.to_csv(os.path.join(OUT_DIR, "Table1_main_metrics.csv"), index=False)
    to_latex(metrics,
             os.path.join(OUT_DIR, "Table1_main_metrics.tex"),
             caption="E. coli: Comparison of LSTM expression scores, controllability, and GC "
                     "content between our end-to-end model and PromoDGDE. "
                     "MWU: Mann–Whitney U test versus Real sequences; "
                     "*, **, *** denote p<0.05, p<0.01, p<0.001.",
             label="tab:ecoli_main")

    kmer = build_kmer_table(df)
    kmer_wide = kmer.pivot_table(index=["k", "Condition"], columns="Method",
                                  values="Pearson r").reset_index()
    kmer_wide.columns.name = None
    kmer_wide.to_csv(os.path.join(OUT_DIR, "Table2_kmer.csv"), index=False)
    to_latex(kmer_wide,
             os.path.join(OUT_DIR, "Table2_kmer.tex"),
             caption="E. coli: k-mer distributional fidelity (Pearson r) for k=3,4,5.",
             label="tab:ecoli_kmer")

    print("\n── Table 1: Main metrics ──")
    print(metrics.to_string(index=False))
    print("\n── Table 2: k-mer fidelity ──")
    print(kmer_wide.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 12.  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Publication-quality comparison (E. coli): our vs PromoDGDE")
    print("=" * 60)

    Q1, Q2 = compute_thresholds(REAL_CSV)
    TR = {"low": (-np.inf, Q1), "mid": (Q1, Q2), "high": (Q2, np.inf)}
    print(f"\nThresholds  Q1={Q1:.4f}  Q2={Q2:.4f}")
    print(f"  low  : score ≤ {Q1:.4f}")
    print(f"  mid  : {Q1:.4f} < score ≤ {Q2:.4f}")
    print(f"  high : score > {Q2:.4f}")

    print("\n[1/4] Loading sequences …")
    real_df = load_real(REAL_CSV, Q1, Q2)
    gen_df  = load_generated()
    all_df  = pd.concat([real_df, gen_df], ignore_index=True)

    print("\nRaw counts:")
    print(all_df.groupby(["method", "label"]).size().to_string())
    all_df = downsample(all_df, n=N_PER_BIN)
    print("\nAfter downsampling:")
    print(all_df.groupby(["method", "label"]).size().to_string())

    print("\n[2/4] LSTM scoring …")
    all_df = attach_predictor_scores(all_df)

    print("\n=== Score distribution check ===")
    for method in ["our", "PromoDGDE", "Real"]:
        sub = all_df[all_df["method"] == method]["oracle_score"].dropna()
        print(f"{method:12s}: mean={sub.mean():.3f}  std={sub.std():.3f}  "
              f"min={sub.min():.3f}  max={sub.max():.3f}")

    all_df["gc_val"] = all_df["sequence"].apply(gc)
    all_df.to_csv(os.path.join(OUT_DIR, "supplement_all_scores.csv"), index=False)

    print("\n[3/4] Plotting figures …")
    fig_violin(all_df, TR)
    fig_success_and_diversity(all_df, TR)   # 图2：左success rate，右edit distance
    fig_gc_kmer(all_df)
    fig_heatmap(all_df, TR)

    print("\n[4/4] Building tables …")
    make_tables(all_df, TR)

    print("\n" + "=" * 60)
    print(f"Done. All outputs → {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()