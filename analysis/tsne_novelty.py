import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
# =========================
# 静音设置
# =========================
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# =========================
# 全局绘图风格（Nature-like）
# =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False
})

np.random.seed(42)

# =========================
# 路径配置
# =========================
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/high.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/defined_SC_Ura_core80_812.csv"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/analysis_embedding_novelty1"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EMBED_PNG = os.path.join(OUT_DIR, "umap_or_tsne_real_generated.png")
OUT_EMBED_CSV = os.path.join(OUT_DIR, "embedding_points.csv")
OUT_NOVELTY_PNG = os.path.join(OUT_DIR, "novelty_hamming_box.png")
OUT_NOVELTY_CSV = os.path.join(OUT_DIR, "novelty_summary.csv")

# =========================
# 参数
# =========================
LABEL_ORDER = ["low", "mid", "high"]
SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]

# 嵌入参数
METHOD = "umap"       # "umap" 或 "tsne"
K_LIST = [3, 4]       # 用 3-mer + 4-mer 频率构建高维特征
MAX_REAL_PER_BIN = 2000
MAX_GEN_PER_BIN = 1000
RANDOM_SEED = 42

# 新颖性参数
# 若 REAL_CSV 有 split 列，则 novelty 默认优先用 train 作为参考集合
NOVELTY_USE_TRAIN_ONLY_IF_AVAILABLE = True
MAX_REAL_REF_PER_BIN = 3000
MAX_REAL_QUERY_PER_BIN = 800
HAMMING_BATCH_SIZE = 128

# =========================
# 基础函数
# =========================
def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ATCG"])

def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")

def read_txt_sequences(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件: {path}")
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if seq:
                seqs.append(seq)
    if not seqs:
        raise ValueError(f"文件中无有效序列: {path}")
    return seqs

def sample_sequences(seqs, max_n, seed=42):
    seqs = list(seqs)
    if len(seqs) <= max_n:
        return seqs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=max_n, replace=False)
    return [seqs[i] for i in idx]

def filter_same_length(seqs):
    seqs = [standardize_seq(s) for s in seqs]
    seqs = [s for s in seqs if len(s) > 0]
    if len(seqs) == 0:
        return []
    lengths = pd.Series([len(s) for s in seqs])
    main_len = int(lengths.mode().iloc[0])
    return [s for s in seqs if len(s) == main_len]

def unique_keep_order(seqs):
    seen = set()
    out = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# =========================
# 数据加载
# =========================
def load_generated_groups():
    groups = {
        "low": read_txt_sequences(GEN_LOW_TXT),
        "mid": read_txt_sequences(GEN_MID_TXT),
        "high": read_txt_sequences(GEN_HIGH_TXT),
    }
    for k in groups:
        groups[k] = unique_keep_order(filter_same_length(groups[k]))
    return groups

def load_real_df():
    if not os.path.exists(REAL_CSV):
        raise FileNotFoundError(f"未找到真实数据文件: {REAL_CSV}")

    df = pd.read_csv(REAL_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("真实 CSV 中缺少 label 列")

    seq_col = find_seq_col(df)

    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["sequence_std"] = df[seq_col].astype(str).apply(standardize_seq)

    df = df[df["label"].isin(LABEL_ORDER)].copy()
    df = df[df["sequence_std"].str.len() > 0].reset_index(drop=True)

    # 只保留主长度
    lengths = df["sequence_std"].str.len()
    main_len = int(lengths.mode().iloc[0])
    df = df[df["sequence_std"].str.len() == main_len].reset_index(drop=True)

    return df

def build_real_groups(df_real):
    groups = {}
    for label in LABEL_ORDER:
        seqs = df_real.loc[df_real["label"] == label, "sequence_std"].tolist()
        groups[label] = unique_keep_order(seqs)
    return groups

def build_real_groups_for_novelty(df_real):
    if ("split" in df_real.columns) and NOVELTY_USE_TRAIN_ONLY_IF_AVAILABLE:
        train_df = df_real[df_real["split"].astype(str).str.lower() == "train"].copy()
        if len(train_df) > 0:
            df_ref = train_df
        else:
            df_ref = df_real
    else:
        df_ref = df_real

    groups = {}
    for label in LABEL_ORDER:
        seqs = df_ref.loc[df_ref["label"] == label, "sequence_std"].tolist()
        groups[label] = unique_keep_order(seqs)
    return groups

# =========================
# k-mer 特征
# =========================
def build_kmer_vocab(k):
    kmers = ["".join(p) for p in product("ATCG", repeat=k)]
    return kmers, {km: i for i, km in enumerate(kmers)}

def seq_to_kmer_freq(seq, k, kmer_to_idx):
    dim = len(kmer_to_idx)
    vec = np.zeros(dim, dtype=np.float32)
    n = len(seq) - k + 1
    if n <= 0:
        return vec

    valid_count = 0
    for i in range(n):
        km = seq[i:i+k]
        if km in kmer_to_idx:
            vec[kmer_to_idx[km]] += 1.0
            valid_count += 1

    if valid_count > 0:
        vec /= valid_count
    return vec

def build_feature_matrix(seqs, k_list=(3, 4)):
    seqs = [standardize_seq(s) for s in seqs]
    seqs = [s for s in seqs if len(s) > 0]

    vocab_info = {}
    total_dim = 0
    for k in k_list:
        kmers, kmer_to_idx = build_kmer_vocab(k)
        vocab_info[k] = (kmers, kmer_to_idx)
        total_dim += len(kmers)

    X = np.zeros((len(seqs), total_dim), dtype=np.float32)

    for i, seq in enumerate(seqs):
        offset = 0
        for k in k_list:
            _, kmer_to_idx = vocab_info[k]
            vec = seq_to_kmer_freq(seq, k, kmer_to_idx)
            X[i, offset:offset + len(vec)] = vec
            offset += len(vec)

    return X

# =========================
# 降维
# =========================
def reduce_to_2d(X, method="umap", random_seed=42):
    X_scaled = StandardScaler().fit_transform(X)

    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.25,
                metric="euclidean",
                random_state=random_seed
            )
            emb = reducer.fit_transform(X_scaled)
            return emb, "UMAP"
        except Exception:
            pass

    reducer = TSNE(
        n_components=2,
        perplexity=35,
        metric="euclidean",
        init="pca",
        learning_rate="auto",
        random_state=random_seed
    )
    emb = reducer.fit_transform(X_scaled)
    return emb, "t-SNE"

# =========================
# 构建嵌入 dataframe
# =========================
def build_embedding_dataframe(real_groups, gen_groups):
    rows = []

    for label in LABEL_ORDER:
        real_seqs = sample_sequences(real_groups[label], MAX_REAL_PER_BIN, seed=RANDOM_SEED)
        gen_seqs = sample_sequences(gen_groups[label], MAX_GEN_PER_BIN, seed=RANDOM_SEED)

        for s in real_seqs:
            rows.append({
                "sequence_std": s,
                "label": label,
                "source": "Real"
            })
        for s in gen_seqs:
            rows.append({
                "sequence_std": s,
                "label": label,
                "source": "Generated"
            })

    emb_df = pd.DataFrame(rows)

    X = build_feature_matrix(emb_df["sequence_std"].tolist(), k_list=K_LIST)
    coords, used_method = reduce_to_2d(X, method=METHOD, random_seed=RANDOM_SEED)

    emb_df["dim1"] = coords[:, 0]
    emb_df["dim2"] = coords[:, 1]
    emb_df["method"] = used_method
    return emb_df

# =========================
# 绘制嵌入图
# =========================
def plot_embedding_density_panels(emb_df, out_path, max_gen_show=500):
    """
    每个条件一个 panel：
    - real: 2D density / hexbin 背景
    - generated: 散点前景
    """
    color_map = {
        "low": "#1f77b4",
        "mid": "#f0ad4e",
        "high": "#d62728",
    }

    method_name = emb_df["method"].iloc[0]

    x_all = emb_df["dim1"].values
    y_all = emb_df["dim2"].values
    xpad = 0.05 * (x_all.max() - x_all.min())
    ypad = 0.05 * (y_all.max() - y_all.min())
    xlim = (x_all.min() - xpad, x_all.max() + xpad)
    ylim = (y_all.min() - ypad, y_all.max() + ypad)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2), sharex=True, sharey=True)

    rng = np.random.default_rng(42)

    # 尝试用 scipy 做 KDE；没有就退回 hexbin
    try:
        from scipy.stats import gaussian_kde
        use_kde = True
    except Exception:
        use_kde = False

    for ax, label in zip(axes, ["low", "mid", "high"]):
        real_sub = emb_df[(emb_df["source"] == "Real") & (emb_df["label"] == label)].copy()
        gen_sub  = emb_df[(emb_df["source"] == "Generated") & (emb_df["label"] == label)].copy()

        this_color = color_map[label]
        cmap = LinearSegmentedColormap.from_list(
            f"{label}_cmap", ["#ffffff", this_color]
        )

        # ===== 背景：真实序列密度 =====
        if use_kde and len(real_sub) >= 50:
            x = real_sub["dim1"].values
            y = real_sub["dim2"].values

            xx, yy = np.meshgrid(
                np.linspace(*xlim, 140),
                np.linspace(*ylim, 140)
            )
            values = np.vstack([x, y])
            kde = gaussian_kde(values)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            positive = zz[zz > 0]
            if len(positive) > 0:
                levels = np.quantile(positive, [0.55, 0.72, 0.84, 0.92, 0.97])
                levels = np.unique(levels)
                if len(levels) >= 2:
                    ax.contourf(xx, yy, zz, levels=np.r_[positive.min(), levels, positive.max()],
                                cmap=cmap, alpha=0.85)
                    ax.contour(xx, yy, zz, levels=levels,
                               colors=this_color, linewidths=0.6, alpha=0.9)
                else:
                    ax.hexbin(
                        x, y,
                        gridsize=35,
                        cmap=cmap,
                        mincnt=1,
                        linewidths=0,
                        alpha=0.75
                    )
            else:
                ax.hexbin(
                    x, y,
                    gridsize=35,
                    cmap=cmap,
                    mincnt=1,
                    linewidths=0,
                    alpha=0.75
                )
        else:
            ax.hexbin(
                real_sub["dim1"], real_sub["dim2"],
                gridsize=35,
                cmap=cmap,
                mincnt=1,
                linewidths=0,
                alpha=0.75
            )

        # ===== 前景：生成序列散点 =====
        if len(gen_sub) > max_gen_show:
            idx = rng.choice(len(gen_sub), size=max_gen_show, replace=False)
            gen_show = gen_sub.iloc[idx].copy()
        else:
            gen_show = gen_sub.copy()

        ax.scatter(
            gen_show["dim1"], gen_show["dim2"],
            s=22,
            c=this_color,
            marker="^",
            alpha=0.90,
            edgecolors="black",
            linewidths=0.25
        )

        ax.set_title(label, pad=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"{method_name}-2")
    for ax in axes:
        ax.set_xlabel(f"{method_name}-1")

    legend_handles = [
        Line2D([0], [0], marker='s', color='none',
               markerfacecolor='lightgray', markeredgecolor='lightgray',
               markersize=8, label='Real density'),
        Line2D([0], [0], marker='^', color='none',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=7, label='Generated sequences')
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False
    )

    fig.suptitle(f"{method_name} embedding: real density and generated sequences", y=1.06, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()

# =========================
# Hamming 新颖性分析
# =========================
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}

def encode_sequences_to_int_array(seqs):
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=np.uint8)

    seqs = filter_same_length(seqs)
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=np.uint8)

    L = len(seqs[0])
    arr = np.zeros((len(seqs), L), dtype=np.uint8)

    for i, seq in enumerate(seqs):
        arr[i] = np.array([BASE_TO_INT[ch] for ch in seq], dtype=np.uint8)

    return arr

def min_hamming_distance(query_arr, ref_arr, batch_size=128):
    """
    返回 query 中每条序列到 ref 集合的最小 Hamming 距离（按位点数，不是比例）
    """
    if len(query_arr) == 0 or len(ref_arr) == 0:
        return np.array([], dtype=float)

    L = query_arr.shape[1]
    out = []

    for start in range(0, len(query_arr), batch_size):
        q = query_arr[start:start+batch_size]  # [b, L]
        # [b, n_ref, L]
        diff = (q[:, None, :] != ref_arr[None, :, :])
        dist = diff.sum(axis=2)  # [b, n_ref]
        min_dist = dist.min(axis=1)
        out.append(min_dist.astype(float))

    return np.concatenate(out)

def compute_novelty(real_groups_ref, gen_groups):
    rows = []
    dist_rows = []

    rng = np.random.default_rng(RANDOM_SEED)

    for label in LABEL_ORDER:
        real_ref_full = unique_keep_order(filter_same_length(real_groups_ref[label]))
        gen_full = unique_keep_order(filter_same_length(gen_groups[label]))

        # exact overlap
        real_ref_set = set(real_ref_full)
        exact_match_count = sum([1 for s in gen_full if s in real_ref_set])
        exact_match_rate = exact_match_count / max(len(gen_full), 1)

        # 采样 reference
        real_ref = sample_sequences(real_ref_full, MAX_REAL_REF_PER_BIN, seed=RANDOM_SEED)
        gen_q = list(gen_full)

        # 构造 real-query（用于 baseline: real -> real）
        if len(real_ref_full) > 10:
            ref_set = set(real_ref)
            real_query_pool = [s for s in real_ref_full if s not in ref_set]
            if len(real_query_pool) < 50:
                # 若去掉 ref 后太少，则从 full 中随机抽，但允许与 ref 有重复
                real_query_pool = list(real_ref_full)
        else:
            real_query_pool = list(real_ref_full)

        real_q = sample_sequences(real_query_pool, MAX_REAL_QUERY_PER_BIN, seed=RANDOM_SEED + 1)

        # 编码
        ref_arr = encode_sequences_to_int_array(real_ref)
        gen_arr = encode_sequences_to_int_array(gen_q)
        real_q_arr = encode_sequences_to_int_array(real_q)

        if (len(ref_arr) == 0) or (len(gen_arr) == 0) or (len(real_q_arr) == 0):
            continue

        gen_min_dist = min_hamming_distance(gen_arr, ref_arr, batch_size=HAMMING_BATCH_SIZE)
        real_min_dist = min_hamming_distance(real_q_arr, ref_arr, batch_size=HAMMING_BATCH_SIZE)

        # summary
        rows.append({
            "bin_level": label,
            "n_generated_unique": len(gen_full),
            "n_real_reference_unique": len(real_ref_full),
            "exact_match_count": exact_match_count,
            "exact_match_rate": exact_match_rate,
            "gen_to_real_min_hamming_mean": float(np.mean(gen_min_dist)),
            "gen_to_real_min_hamming_median": float(np.median(gen_min_dist)),
            "gen_to_real_min_hamming_q25": float(np.quantile(gen_min_dist, 0.25)),
            "gen_to_real_min_hamming_q75": float(np.quantile(gen_min_dist, 0.75)),
            "real_to_real_min_hamming_mean": float(np.mean(real_min_dist)),
            "real_to_real_min_hamming_median": float(np.median(real_min_dist)),
            "real_to_real_min_hamming_q25": float(np.quantile(real_min_dist, 0.25)),
            "real_to_real_min_hamming_q75": float(np.quantile(real_min_dist, 0.75)),
        })

        for v in gen_min_dist:
            dist_rows.append({
                "bin_level": label,
                "source": "Generated→Real",
                "min_hamming": float(v)
            })
        for v in real_min_dist:
            dist_rows.append({
                "bin_level": label,
                "source": "Real→Real",
                "min_hamming": float(v)
            })

    summary_df = pd.DataFrame(rows)
    dist_df = pd.DataFrame(dist_rows)
    return summary_df, dist_df

def _ecdf(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    vals = np.sort(vals)
    y = np.arange(1, len(vals) + 1) / len(vals)
    return vals, y

def plot_novelty_ecdf_panels(dist_df, summary_df, out_path):
    """
    每个条件一个 panel：
    - Real→Real vs Generated→Real 的最近邻 Hamming 距离 ECDF
    - exact rate 写到 panel 标题中
    """
    color_map = {
        "Real→Real": "#4C72B0",
        "Generated→Real": "#DD8452",
    }

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0), sharey=True)

    for ax, label in zip(axes, ["low", "mid", "high"]):
        sub = dist_df[dist_df["bin_level"] == label].copy()
        sum_sub = summary_df[summary_df["bin_level"] == label].copy()

        if len(sum_sub) > 0:
            exact_rate = float(sum_sub["exact_match_rate"].iloc[0]) * 100
        else:
            exact_rate = np.nan

        medians = {}

        for source in ["Real→Real", "Generated→Real"]:
            vals = sub.loc[sub["source"] == source, "min_hamming"].dropna().values
            if len(vals) == 0:
                continue

            x, y = _ecdf(vals)
            ax.step(
                x, y,
                where="post",
                linewidth=1.7,
                color=color_map[source],
                label=source
            )

            med = float(np.median(vals))
            medians[source] = med
            ax.axvline(
                med,
                linestyle="--",
                linewidth=0.9,
                color=color_map[source],
                alpha=0.8
            )

        title = f"{label}\nexact={exact_rate:.2f}%"
        ax.set_title(title, pad=5)

        if len(medians) == 2:
            ax.text(
                0.97, 0.05,
                f"median: {medians['Real→Real']:.0f} vs {medians['Generated→Real']:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6
            )

        ax.set_xlabel("Nearest-neighbor Hamming distance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Cumulative fraction")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False
    )

    fig.suptitle("Novelty relative to real reference set", y=1.12, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
# =========================
# 主函数
# =========================
def main():
    # 1) 读数据
    gen_groups = load_generated_groups()
    real_df = load_real_df()
    real_groups = build_real_groups(real_df)
    real_groups_ref = build_real_groups_for_novelty(real_df)

    # 2) 2D embedding
    emb_df = build_embedding_dataframe(real_groups, gen_groups)
    emb_df.to_csv(OUT_EMBED_CSV, index=False)
    plot_embedding_density_panels(emb_df, OUT_EMBED_PNG, max_gen_show=450)

    # 3) novelty
    novelty_summary_df, novelty_dist_df = compute_novelty(real_groups_ref, gen_groups)
    novelty_summary_df.to_csv(OUT_NOVELTY_CSV, index=False)
    plot_novelty_ecdf_panels(novelty_dist_df, novelty_summary_df, OUT_NOVELTY_PNG)

    # 4) 打印简要结果
    print("=" * 80)
    print(f"Embedding plot saved to: {OUT_EMBED_PNG}")
    print(f"Embedding coordinates saved to: {OUT_EMBED_CSV}")
    print(f"Novelty plot saved to: {OUT_NOVELTY_PNG}")
    print(f"Novelty summary saved to: {OUT_NOVELTY_CSV}")
    print("=" * 80)

    if len(novelty_summary_df) > 0:
        print("\nNovelty summary:")
        print(novelty_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()