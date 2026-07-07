import os, sys, re, warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import Levenshtein
from itertools import combinations
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 0. Nature / NPG 主题（字体统一放大，确保论文中可读）
# ══════════════════════════════════════════════════════════════
NATURE_RC = {
    "font.size":           11,    # 基础字体 10→11
    "axes.labelsize":      11,    # 轴标签
    "axes.titlesize":      11,    # 子图标题
    "xtick.labelsize":     10,    # x 刻度
    "ytick.labelsize":     10,    # y 刻度
    "legend.fontsize":     9,     # 图例
    "legend.frameon":      False,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "axes.unicode_minus":   False,
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial"],
    "font.weight":          "normal",
    "axes.labelweight":     "normal",
    "axes.titleweight":     "normal",
}

# ══════════════════════════════════════════════════════════════
# ── 参考图配色（学术暖色系，低饱和度）
# ══════════════════════════════════════════════════════════════
C_REAL = "#B8B8B8"   # 灰色（Real）
C_OUR  = "#E8B8B8"   # 玫粉（ConProDiff）
C_PRO  = "#A8BDD4"   # 灰蓝（PromoDGDE）

# Fig E 柱状图——每个 label 的 PromoDGDE / ConProDiff 颜色直接用上面三色
# Fig F 箱线图同上

# Fig C t-SNE density 配色保持原蓝紫系不变（t-SNE 是独立子图）
C_LOW  = "#A3C4E0"
C_MID  = "#A99ED1"
C_HIGH = "#7B7FB8"
STRENGTH_COLORS = {"low": C_LOW, "mid": C_MID, "high": C_HIGH}

FIG_W = 3.46
FIG_H = 3.10

LABEL_ORDER   = ["low", "mid", "high"]
LABEL_DISPLAY = {"low": "Low", "mid": "Mid", "high": "High"}

# ══════════════════════════════════════════════════════════════
# 1. 路径配置
# ══════════════════════════════════════════════════════════════
GEN_LOW_TXT   = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/low.txt"
GEN_MID_TXT   = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/high.txt"

PROMODGDE_LOW_TXT   = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/low_final_sequences_ecoli.txt"
PROMODGDE_MID_TXT   = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/medium_final_sequences_ecoli.txt"
PROMODGDE_HIGH_TXT = "/home/yt/Code/PromoDGDE/Optimizer/results_opt/ecoli_165bp/high_final_sequences_ecoli.txt"

REAL_CSV      = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
TRAIN_CSV     = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp.csv"
PREDICTOR_DIR = "/home/yt/Code/DNA-Diffusion/Predictor"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/lunwen/tu2_ecoli"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 2. 工具函数
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

def compute_thresholds(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next(c for c in ["strength","expression","exp"] if c in df.columns)
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(1/3)), float(vals.quantile(2/3))

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

def min_edit_to_set(query_seqs, ref_seqs, trunc=100, n_ref=300, seed=42):
    rng = np.random.RandomState(seed)
    ref = rng.choice(ref_seqs, min(n_ref, len(ref_seqs)), replace=False).tolist()
    ref_t = [r[:trunc] for r in ref]
    dists = []
    for q in query_seqs:
        qs = q[:trunc]
        d = min(Levenshtein.distance(qs, r) / max(len(qs), len(r), 1)
                for r in ref_t)
        dists.append(d)
    return np.array(dists)

def kmer_freq(seqs, k):
    counts = Counter()
    total  = 0
    for s in seqs:
        for i in range(len(s)-k+1):
            counts[s[i:i+k]] += 1
            total += 1
    return Counter({km: v/total for km, v in counts.items()}) if total else counts

def kmer_pearson(seqs_a, seqs_b, k):
    fa, fb = kmer_freq(seqs_a, k), kmer_freq(seqs_b, k)
    keys = sorted(set(fa)|set(fb))
    va = np.array([fa.get(kk,0) for kk in keys])
    vb = np.array([fb.get(kk,0) for kk in keys])
    if va.std()<1e-10 or vb.std()<1e-10: return 0.0
    return float(np.corrcoef(va, vb)[0,1])

def gc_content(seqs):
    return np.array([(s.count("G")+s.count("C"))/len(s)
                     for s in seqs if len(s)>0])

def seq_to_onehot_flat(seqs, max_len):
    mapping = {"A":0,"C":1,"G":2,"T":3}
    arr = np.zeros((len(seqs), max_len*4), dtype=np.float32)
    for i,s in enumerate(seqs):
        for j,ch in enumerate(s[:max_len]):
            if ch in mapping:
                arr[i, j*4+mapping[ch]] = 1.0
    return arr

# ══════════════════════════════════════════════════════════════
# 3. 数据加载
# ══════════════════════════════════════════════════════════════
print("="*58)
print(" Loading sequences")
print("="*58)

Q1, Q2   = compute_thresholds(REAL_CSV)
real_df  = load_real_df(REAL_CSV, Q1, Q2)
print(f"   Thresholds Q1={Q1:.4f} Q2={Q2:.4f}  |  Real: {len(real_df)}")

gen_seqs = {
    "our":      {"low": read_txt(GEN_LOW_TXT),
                 "mid": read_txt(GEN_MID_TXT),
                 "high":read_txt(GEN_HIGH_TXT)},
    "PromoDGDE":{"low": read_txt(PROMODGDE_LOW_TXT),
                 "mid": read_txt(PROMODGDE_MID_TXT),
                 "high":read_txt(PROMODGDE_HIGH_TXT)},
}
for m,d in gen_seqs.items():
    for lbl,s in d.items():
        print(f"  {m:12s} | {lbl}: n={len(s)}")

METHOD_META = [
    ("PromoDGDE", C_PRO, "PromoDGDE"),
    ("our",       C_OUR, "ConProDiff"),
]
GROUP_POS = {"low":1.0, "mid":2.0, "high":3.0}

# ══════════════════════════════════════════════════════════════
# Fig C — t-SNE
# ══════════════════════════════════════════════════════════════
def plot_fig_c(save_stem="FigC_tsne_ec"):
    print("\n  [Fig C] t-SNE — Real KDE density + Generated triangles …")
    from itertools import product as iproduct
    import matplotlib.lines as mlines

    ALL_KMERS = ["".join(p) for p in iproduct("ACGT", repeat=4)]
    KMER_IDX  = {km: i for i, km in enumerate(ALL_KMERS)}

    def to_kmer_mat(seqs, max_len=165, k=4):
        mat = np.zeros((len(seqs), len(ALL_KMERS)), dtype=np.float32)
        for i, s in enumerate(seqs):
            s = s[:max_len]
            total = len(s) - k + 1
            if total <= 0:
                continue
            for j in range(total):
                km = s[j:j+k]
                if km in KMER_IDX:
                    mat[i, KMER_IDX[km]] += 1
            mat[i] /= total
        return mat

    COND_PALETTE = {
        "low": {
            "fill_light": "#DCE9F5", "fill_mid":   "#ADC9E8", "fill_dark":  "#6FA8D4",
            "line":       "#4A86B8", "marker":     "#2E5F8A",
        },
        "mid": {
            "fill_light": "#E4DFF4", "fill_mid":   "#C0B4E4", "fill_dark":  "#8F7FCB",
            "line":       "#6B5BAD", "marker":     "#4A3A82",
        },
        "high": {                                        # 改为玫粉系
            "fill_light": "#F5DDE0", "fill_mid": "#E8AABA", "fill_dark": "#D4758A",
            "line":       "#B84D65", "marker":   "#8A2A42",
    },
    }

    rng = np.random.RandomState(42)
    N_REAL = 300
    N_GEN  = 250

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(FIG_W * 2, FIG_H * 0.82), 
            constrained_layout=True,
        )
        
        # 恢复正常的紧凑布局间距，没有任何顶部额外留白
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0, wspace=0)

        for ax, lbl in zip(axes, LABEL_ORDER):
            pal = COND_PALETTE[lbl]

            real_pool = real_df.loc[real_df["label"] == lbl, "sequence"].tolist()
            real_s = rng.choice(real_pool, min(N_REAL, len(real_pool)), replace=False).tolist()
            gen_s = rng.choice(gen_seqs["our"][lbl], min(N_GEN, len(gen_seqs["our"][lbl])), replace=False).tolist()

            all_s = real_s + gen_s
            src   = ["real"] * len(real_s) + ["gen"] * len(gen_s)
            X     = to_kmer_mat(all_s)
            npca  = min(40, X.shape[1], X.shape[0] - 1)
            Xp    = PCA(n_components=npca, random_state=42).fit_transform(X)
            Xt    = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1200, verbose=0).fit_transform(Xp)

            pts_real = Xt[[i for i, l in enumerate(src) if l == "real"]]
            pts_gen  = Xt[[i for i, l in enumerate(src) if l == "gen"]]

            grid_n = 150
            x_all, y_all = Xt[:, 0], Xt[:, 1]
            margin = (np.ptp(x_all) + np.ptp(y_all)) * 0.06
            xi = np.linspace(x_all.min() - margin, x_all.max() + margin, grid_n)
            yi = np.linspace(y_all.min() - margin, y_all.max() + margin, grid_n)
            Xi, Yi = np.meshgrid(xi, yi)
            grid   = np.vstack([Xi.ravel(), Yi.ravel()])

            if len(pts_real) >= 10:
                kde = gaussian_kde(pts_real.T, bw_method=0.30)
                Zi  = kde(grid).reshape(grid_n, grid_n)

                levels = 4
                ax.contourf(Xi, Yi, Zi, levels=levels, colors=[pal["fill_light"]] * (levels - 1), alpha=0.55, zorder=1)
                q75 = np.percentile(Zi, 75)
                q90 = np.percentile(Zi, 90)
                ax.contourf(Xi, Yi, Zi, levels=[q75, Zi.max()], colors=[pal["fill_mid"]], alpha=0.50, zorder=2)
                ax.contourf(Xi, Yi, Zi, levels=[q90, Zi.max()], colors=[pal["fill_dark"]], alpha=0.45, zorder=3)
                ax.contour(Xi, Yi, Zi, levels=levels, colors=[pal["line"]], linewidths=0.7, alpha=0.60, zorder=4)

            ax.scatter(
                pts_gen[:, 0], pts_gen[:, 1],
                marker="^", c=pal["marker"], s=14, alpha=0.72,
                edgecolors="white", linewidths=0.25, rasterized=True, zorder=5,
                label="Generated sequences",
            )

            # 恢复干净美观的原本子图标题
            ax.set_title(LABEL_DISPLAY[lbl], fontsize=11, pad=5, y=1.06,color="#2A2A3A")
            
            ax.set_xlabel("t-SNE 1", fontsize=9, color="#555566")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_facecolor("white")

            if ax is axes[0]:
                ax.set_ylabel("t-SNE 2", fontsize=9, color="#555566")
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_linewidth(0.6)
                ax.spines["left"].set_color("#AAAACC")

        # ── 核心改动：把图例放进最右侧子图 axes[2] 的坐标系内部右上角 ──
        density_patch = Patch(facecolor="#B8C4DE", edgecolor="#7A88BB", alpha=0.65, linewidth=0.7, label="Real density")
        gen_marker = mlines.Line2D([], [], marker="^", color="black", markersize=6, linewidth=0, label="Generated sequences")
        
        fig.legend(
            handles=[density_patch, gen_marker],
            loc="upper center",   # 自动锁定在 axes[2] 绘图区框线内侧的右上方
            bbox_to_anchor=(0.5, 0.96),  # 轻微向上调整，确保与子图标题有适当间距
            ncol=2,              # 单列上下排布，完美融入图内空白
            fontsize=9,
            frameon=False,       # 无边框
            handletextpad=0.4,
        )

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"), dpi=300, bbox_inches="tight")
            print(f"   Saved {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Fig D — 组内编辑距离分布小提琴图（修改背景为白，无虚线）
# ══════════════════════════════════════════════════════════════
def intra_dist_all(seqs, trunc=100, n_sample=120, seed=42):
    rng = np.random.RandomState(seed)
    s = [x[:trunc] for x in rng.choice(seqs, min(n_sample, len(seqs)), replace=False).tolist()]
    dists = [Levenshtein.distance(a, b) / max(len(a), len(b), 1) for a, b in combinations(s, 2)]
    return dists

def plot_fig_d(save_stem="FigD_diversity_ec"):
    print("\n  [Fig D] Intra-group diversity violin (White background) …")

    dist_data = {}
    for method in ["our", "PromoDGDE"]:
        dist_data[method] = {}
        for lbl in LABEL_ORDER:
            dist_data[method][lbl] = intra_dist_all(gen_seqs[method][lbl])

    # 加入真实序列的组内编辑距离
    dist_data["real"] = {}
    for lbl in LABEL_ORDER:
        real_seqs_lbl = real_df.loc[real_df["label"] == lbl, "sequence"].tolist()
        dist_data["real"][lbl] = intra_dist_all(real_seqs_lbl)

    OFFSETS_V = {"real": -0.26, "PromoDGDE": 0.0, "our": 0.26}

    BOX_STYLE = {
        "real":      {"color": "#C8C8C8", "edge": "#888888", "label": "Real"},
        "PromoDGDE": {"color": C_PRO,     "edge": "#6A8EAA", "label": "PromoDGDE"},
        "our":       {"color": C_OUR,     "edge": "#C89090", "label": "ConProDiff"},
    }

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.set_facecolor("white")

        all_d_vals = []

        for method in ["real", "PromoDGDE", "our"]:
            sty = BOX_STYLE[method]
            for i, lbl in enumerate(LABEL_ORDER):
                vals = dist_data[method][lbl]
                if len(vals) < 10:
                    continue

                all_d_vals.extend(vals)
                xpos = (i + 1) + OFFSETS_V[method]

                parts = ax.violinplot(vals, positions=[xpos], widths=0.22,
                                      showmedians=True, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(sty["color"])
                    pc.set_edgecolor(sty["edge"])
                    pc.set_alpha(0.55)
                parts["cmedians"].set_color(sty["edge"])
                parts["cmedians"].set_linewidth(1.4)

        # Y 轴根据数据范围自动设置
        v_min, v_max = np.min(all_d_vals), np.max(all_d_vals)
        margin = (v_max - v_min) * 0.08
        ax.set_ylim(v_min - margin, v_max + margin)

        legend_handles = [
            Patch(facecolor=BOX_STYLE["real"]["color"],      edgecolor=BOX_STYLE["real"]["edge"],      alpha=0.7, label="Real"),
            Patch(facecolor=BOX_STYLE["PromoDGDE"]["color"], edgecolor=BOX_STYLE["PromoDGDE"]["edge"], alpha=0.7, label="PromoDGDE"),
            Patch(facecolor=BOX_STYLE["our"]["color"],       edgecolor=BOX_STYLE["our"]["edge"],       alpha=0.7, label="ConProDiff"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.02, 0))

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_xlim(0.55, 3.45)
        ax.set_ylabel("Intra-group edit distance")
        ax.grid(False)

        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f"   Saved {save_stem}.{ext}")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Fig E — k-mer Pearson 相关系数柱状图（修改背景为白，去除网格线）
# ══════════════════════════════════════════════════════════════
def plot_fig_e(save_stem="FigE_kmer_ec"):
    print("\n  [Fig E] k-mer correlations …")
    
    k = 4
    results = {}
    for method in ["our", "PromoDGDE"]:
        results[method] = {}
        for lbl in LABEL_ORDER:
            real_seqs = real_df.loc[real_df["label"] == lbl, "sequence"].tolist()
            r = kmer_pearson(gen_seqs[method][lbl], real_seqs, k)
            results[method][lbl] = r

    # 每个 method 对应参考图风格配色
    BAR_STYLE = {
        "PromoDGDE": {"color": "#A8BDD4", "edgecolor": "#6A8EAA", "label": "PromoDGDE"},
        "our":       {"color": "#E8B8B8", "edgecolor": "#C89090", "label": "ConProDiff"},
    }

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.set_facecolor("white")

        x = np.arange(len(LABEL_ORDER))
        width = 0.30
        OFFSETS_BAR = {"PromoDGDE": -0.5, "our": 0.5}

        for method, col, label in METHOD_META:
            sty = BAR_STYLE[method]
            vals = [results[method][lbl] for lbl in LABEL_ORDER]
            bars = ax.bar(
                x + OFFSETS_BAR[method] * width,
                vals,
                width=width,
                color=sty["color"],
                alpha=0.85,
                edgecolor=sty["edgecolor"],
                linewidth=0.9,
                label=sty["label"],
                zorder=3,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#333333",
                )

        ax.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_ylabel(f"Pearson ({k}-mer)")
        ax.set_ylim(0.7, 1.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.tick_params(axis="y", which="minor", length=2, width=0.5)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 1.04),
            handlelength=1.0, handletextpad=0.4,
            borderpad=0.5, labelspacing=0.3,
        )

        fig.tight_layout(pad=0.4)
        d = 0.012
        kwargs = dict(transform=ax.transAxes, color="k",
                      linewidth=0.8, clip_on=False)
        ax.plot((-d, +d), (-d*0.6, +d*0.6), **kwargs)                        # 第一条
        ax.plot((-d, +d), (-d*0.6 + 0.022, +d*0.6 + 0.022), **kwargs)        # 第二条，y整体上移0.022

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f"   Saved {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Fig F — GC 含量箱线图（修改背景为白，无虚线）
# ══════════════════════════════════════════════════════════════
def plot_fig_f(save_stem="FigF_gc_ec"):
    print("\n  [Fig F] GC content boxplot …")

    OFFSETS_V = {"real": -0.26, "PromoDGDE": 0.0, "our": 0.26}

    # 参考图配色：Real=灰蓝，PromoDGDE=暖黄，ConProDiff=玫粉
    BOX_STYLE = {
        "real":      {"face": "#C8C8C8", "edge": "#888888", "label": "Real"},
        "PromoDGDE": {"face": "#A8BDD4", "edge": "#6A8EAA", "label": "PromoDGDE"},
        "our":       {"face": "#E8B8B8", "edge": "#C89090", "label": "ConProDiff"},
    }
    ALL_META_F = [("real", None, "Real"), ("PromoDGDE", None, "PromoDGDE"), ("our", None, "ConProDiff")]

    N_MAX = 300
    rng = np.random.RandomState(42)

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.set_facecolor("white")

        all_vals = []

        for method, _, label in ALL_META_F:
            sty = BOX_STYLE[method]
            for lbl in LABEL_ORDER:
                if method == "real":
                    seqs = real_df.loc[real_df["label"] == lbl, "sequence"].tolist()
                else:
                    seqs = gen_seqs[method][lbl]

                seqs = [s for s in seqs if len(s) >= 50]
                if len(seqs) > N_MAX:
                    seqs = rng.choice(seqs, N_MAX, replace=False).tolist()

                vals = gc_content(seqs)
                if len(vals) < 10:
                    continue

                all_vals.extend(vals)
                xpos = GROUP_POS[lbl] + OFFSETS_V[method]
                ax.boxplot(
                    vals,
                    positions=[xpos],
                    widths=0.20,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=sty["edge"], linewidth=1.8),
                    whiskerprops=dict(color=sty["edge"], linewidth=0.9),
                    capprops=dict(color=sty["edge"], linewidth=0.9),
                    boxprops=dict(
                        facecolor=sty["face"],
                        alpha=0.80,          # 提高不透明度，更接近参考图
                        edgecolor=sty["edge"],
                        linewidth=1.0,
                    ),
                )

        v_min, v_max = np.min(all_vals), np.max(all_vals)
        margin = (v_max - v_min) * 0.08
        ax.set_ylim(v_min - margin, v_max + margin)

        ax.set_xticks([GROUP_POS[l] for l in LABEL_ORDER])
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER])
        ax.set_xlim(0.58, 3.42)
        ax.set_ylabel("GC content")
        ax.grid(False)

        legend_handles = [
            Patch(facecolor=BOX_STYLE["real"]["face"],      edgecolor=BOX_STYLE["real"]["edge"],
                  alpha=0.85, linewidth=0.9, label="Real"),
            Patch(facecolor=BOX_STYLE["PromoDGDE"]["face"], edgecolor=BOX_STYLE["PromoDGDE"]["edge"],
                  alpha=0.85, linewidth=0.9, label="PromoDGDE"),
            Patch(facecolor=BOX_STYLE["our"]["face"],       edgecolor=BOX_STYLE["our"]["edge"],
                  alpha=0.85, linewidth=0.9, label="ConProDiff"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.02, 1.00))

        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{save_stem}.{ext}"))
            print(f"   Saved {save_stem}.{ext}")
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════════
print("\n" + "="*58)
print(" Panel 2 — Diversity & Novelty (EC)")
print("="*58)
np.random.seed(42)
plot_fig_c()
plot_fig_d()
plot_fig_e()
plot_fig_f()
print("\nAll figures saved to:", OUT_DIR)
