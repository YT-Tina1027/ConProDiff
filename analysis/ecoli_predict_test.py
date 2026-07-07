import os
import re
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# ================== 静音设置 ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ================== 全局绘图风格（Nature） ==================
plt.rcParams.update({
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

# ================== 路径配置 ==================
REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/ecoli_test_3000.csv"

PREDICTOR_MODEL_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"

# 这里是你的 predictor_models.py 所在工程根目录
BIO_ROOT = "/home/yt/Code/Bio_diffusion_gan"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/predictor_ecoli_test_3000_quantile"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PRED_CSV = os.path.join(OUT_DIR, "real_sequences_with_pred_strength.csv")
OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "real_vs_pred_strength_summary.csv")
OUT_METRICS_CSV = os.path.join(OUT_DIR, "predictor_metrics.csv")
OUT_BOXPLOT = os.path.join(OUT_DIR, "real_vs_pred_strength_box.png")
OUT_BOXPLOT_PDF = os.path.join(OUT_DIR, "real_vs_pred_strength_box.pdf")
OUT_SCATTER = os.path.join(OUT_DIR, "real_vs_pred_scatter.png")
OUT_SCATTER_PDF = os.path.join(OUT_DIR, "real_vs_pred_scatter.pdf")

# ================== 参数 ==================
SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]

LABEL_ORDER = ["low", "mid", "high"]
SOURCE_ORDER = ["Real", "Predicted"]

# 当前 ecoli_test_3000.csv 没有 split，所以使用 all
EVAL_SPLIT = "all"

BATCH_SIZE = 1024


# ================== 基础函数 ==================
def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join(ch for ch in s if ch in "ATCG")


def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")


def get_full_limits(series, pad_ratio=0.05, min_pad=0.01):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return None

    lo, hi = np.min(vals), np.max(vals)

    if hi > lo:
        pad = max((hi - lo) * pad_ratio, min_pad)
    else:
        pad = min_pad

    return lo - pad, hi + pad


# ================== 数据读取：当前 CSV 内部三分位分箱 ==================
def load_real_df(path: str) -> pd.DataFrame:
    """
    读取 ecoli_test_3000.csv。

    要求输入至少包含：
    - sequence
    - strength

    本函数不读取任何 pkl。
    直接基于当前 CSV 的 strength 重新做三分位分箱：
    - lowest 1/3  -> low
    - middle 1/3  -> mid
    - highest 1/3 -> high
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到真实数据文件: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    seq_col = find_seq_col(df)

    if "strength" not in df.columns:
        raise ValueError("真实数据缺少必要列: strength")

    df = df.copy()

    df["sequence_std"] = df[seq_col].astype(str).apply(standardize_seq)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")

    df = df.dropna(subset=["sequence_std", "strength"]).copy()
    df = df[df["sequence_std"].str.len() > 0].copy()

    if len(df) == 0:
        raise ValueError("过滤后没有有效序列。")

    # =====================================================
    # 基于当前 CSV 的 strength 重新做三分位分箱
    # 不使用 processed_data/ecoli_diffusion_data.pkl
    # =====================================================
    labels_id, bin_edges = pd.qcut(
        df["strength"],
        q=3,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    bin_edges = np.asarray(bin_edges, dtype=np.float32)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    if len(bin_edges) != 4:
        raise ValueError(
            f"三分位分箱失败，期望得到 3 个 bin，但实际得到 {len(bin_edges) - 1} 个。"
            "可能是 strength 重复值太多导致 qcut 合并 bin。"
        )

    id_to_label = {
        0: "low",
        1: "mid",
        2: "high",
    }

    df["label"] = labels_id.astype(int).map(id_to_label)

    # 当前文件没有 split，统一设为 all
    df["split"] = "all"

    df = df[df["label"].isin(LABEL_ORDER)].copy()

    print("[Info] Loaded real evaluation data:")
    print(f"  path = {path}")
    print(f"  n    = {len(df)}")
    print("[Info] Quantile bin edges from current CSV:")
    print(f"  bin_edges = {bin_edges}")
    print(f"  low  <= {bin_edges[1]:.6f}")
    print(f"  mid  = ({bin_edges[1]:.6f}, {bin_edges[2]:.6f}]")
    print(f"  high > {bin_edges[2]:.6f}")
    print("[Info] Label counts:")
    print(df["label"].value_counts().reindex(LABEL_ORDER))

    return df.reset_index(drop=True)


# ================== Predictor 载入与预测 ==================
def load_predictor_and_scaler(real_df: pd.DataFrame):
    """
    载入 E. coli predictor。

    注意：
    这里 scaler 仍然基于当前 real_df 的 strength 拟合。
    如果你的 predictor 训练时使用了单独保存的 scaler，最好改成加载训练时 scaler。
    """
    import torch

    if BIO_ROOT not in sys.path:
        sys.path.insert(0, BIO_ROOT)

    from Predictor.predictor_models import LSTMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(PREDICTOR_MODEL_PATH):
        raise FileNotFoundError(f"predictor 模型不存在: {PREDICTOR_MODEL_PATH}")

    # 当前 test_3000 没有 train split，所以用当前数据拟合 scaler
    # 注意：严格来说，最好使用 predictor 训练时保存的 scaler
    scaler = StandardScaler()
    scaler.fit(real_df["strength"].values.reshape(-1, 1))

    seq_len = int(real_df["sequence_std"].str.len().mode().iloc[0])

    hidden_size = 256
    dropout_rate = 0.2
    lambda_l2 = 0.001

    model = LSTMModel(
        4,
        hidden_size,
        1,
        dropout_rate,
        lambda_l2,
    )

    state = torch.load(PREDICTOR_MODEL_PATH, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("[Info] Predictor loaded:")
    print(f"  model_path = {PREDICTOR_MODEL_PATH}")
    print(f"  device     = {device}")
    print(f"  seq_len    = {seq_len}")

    return model, scaler, seq_len, device


def seq_to_onehot_array(seq: str) -> np.ndarray:
    seq = re.sub("[^ACGTacgt]", "Z", str(seq)).lower().replace("u", "t")
    mapping = {
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3,
    }

    arr = np.zeros((len(seq), 4), dtype=np.float32)

    for i, ch in enumerate(seq):
        if ch in mapping:
            arr[i, mapping[ch]] = 1.0

    return arr


def predict_sequences_with_model(
    sequences,
    model,
    scaler,
    seq_len,
    device,
    batch_size=1024,
):
    import torch

    preds = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        arrs = []

        for s in batch_seqs:
            x = seq_to_onehot_array(s)

            if x.shape[0] < seq_len:
                pad_len = seq_len - x.shape[0]
                x = np.pad(
                    x,
                    ((0, pad_len), (0, 0)),
                    mode="constant",
                )
            elif x.shape[0] > seq_len:
                x = x[:seq_len]

            arrs.append(torch.tensor(x, dtype=torch.float32))

        x = torch.stack(arrs, dim=0).to(device)

        with torch.no_grad():
            y_norm = model(x).detach().cpu().numpy().reshape(-1, 1)
            y_raw = scaler.inverse_transform(y_norm).reshape(-1)

        preds.extend(y_raw.tolist())

    return np.asarray(preds, dtype=float)


# ================== 构建结果表 ==================
def attach_predictions(real_df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, seq_len, device = load_predictor_and_scaler(real_df)

    if EVAL_SPLIT != "all":
        if "split" not in real_df.columns:
            raise ValueError("数据中没有 split 列，但你设置了 EVAL_SPLIT != 'all'")
        eval_df = real_df[real_df["split"] == EVAL_SPLIT].copy()
    else:
        eval_df = real_df.copy()

    if len(eval_df) == 0:
        raise ValueError(f"EVAL_SPLIT={EVAL_SPLIT} 后没有数据。")

    preds = predict_sequences_with_model(
        eval_df["sequence_std"].tolist(),
        model=model,
        scaler=scaler,
        seq_len=seq_len,
        device=device,
        batch_size=BATCH_SIZE,
    )

    eval_df = eval_df.copy()
    eval_df["pred_strength"] = pd.to_numeric(preds, errors="coerce")
    eval_df = eval_df.dropna(subset=["pred_strength", "strength"]).reset_index(drop=True)

    eval_df.to_csv(OUT_PRED_CSV, index=False)

    return eval_df


# ================== 指标计算 ==================
def compute_metrics(eval_df: pd.DataFrame):
    y_true = eval_df["strength"].values
    y_pred = eval_df["pred_strength"].values

    pearson_r = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spearman_rho = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "n": len(eval_df),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "mse": mse,
        "mae": mae,
    }


def compute_metrics_by_label(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in LABEL_ORDER:
        sub = eval_df[eval_df["label"] == label].copy()
        if len(sub) == 0:
            continue

        y_true = sub["strength"].values
        y_pred = sub["pred_strength"].values

        if len(sub) > 1:
            pr = pearsonr(y_true, y_pred)[0]
            sr = spearmanr(y_true, y_pred)[0]
        else:
            pr = np.nan
            sr = np.nan

        rows.append({
            "label": label,
            "n": len(sub),
            "pearson_r": pr,
            "spearman_rho": sr,
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "real_mean": np.mean(y_true),
            "pred_mean": np.mean(y_pred),
            "real_median": np.median(y_true),
            "pred_median": np.median(y_pred),
        })

    return pd.DataFrame(rows)


# ================== 箱线图数据 ==================
def build_strength_plot_df(eval_df: pd.DataFrame) -> pd.DataFrame:
    real_plot = eval_df[["label", "strength"]].copy()
    real_plot["source"] = "Real"
    real_plot["bin_level"] = real_plot["label"]
    real_plot = real_plot.rename(columns={"strength": "value"})

    pred_plot = eval_df[["label", "pred_strength"]].copy()
    pred_plot["source"] = "Predicted"
    pred_plot["bin_level"] = pred_plot["label"]
    pred_plot = pred_plot.rename(columns={"pred_strength": "value"})

    plot_df = pd.concat([real_plot, pred_plot], ignore_index=True)

    plot_df["bin_level"] = pd.Categorical(
        plot_df["bin_level"],
        categories=LABEL_ORDER,
        ordered=True,
    )

    plot_df["source"] = pd.Categorical(
        plot_df["source"],
        categories=SOURCE_ORDER,
        ordered=True,
    )

    return plot_df


def summarize(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = plot_df.groupby(["bin_level", "source"], observed=True)["value"]

    for (bin_level, source), vals in grouped:
        vals = pd.to_numeric(vals, errors="coerce").dropna()

        if len(vals) == 0:
            continue

        rows.append({
            "bin_level": bin_level,
            "source": source,
            "count": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "q25": vals.quantile(0.25),
            "median": vals.median(),
            "q75": vals.quantile(0.75),
            "max": vals.max(),
        })

    return pd.DataFrame(rows)


# ================== 绘图 ==================
def plot_strength_box(plot_df: pd.DataFrame, out_png: str, out_pdf: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))

    pos_map = {
        ("low", "Real"): 0.85,
        ("low", "Predicted"): 1.15,
        ("mid", "Real"): 1.85,
        ("mid", "Predicted"): 2.15,
        ("high", "Real"): 2.85,
        ("high", "Predicted"): 3.15,
    }

    color_map = {
        "Real": "#0072B2",
        "Predicted": "#D55E00",
    }

    for b in LABEL_ORDER:
        for s in SOURCE_ORDER:
            vals = plot_df[
                (plot_df["bin_level"] == b) &
                (plot_df["source"] == s)
            ]["value"].dropna().values

            if len(vals) == 0:
                continue

            ax.boxplot(
                vals,
                positions=[pos_map[(b, s)]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=color_map[s],
                    alpha=0.3,
                    edgecolor=color_map[s],
                    linewidth=1.2,
                ),
                medianprops=dict(
                    color=color_map[s],
                    linewidth=1.5,
                ),
                whiskerprops=dict(
                    color=color_map[s],
                    linewidth=0.8,
                ),
                capprops=dict(
                    color=color_map[s],
                    linewidth=0.8,
                ),
            )

            n_show = min(len(vals), 80)

            if n_show:
                idx = np.random.choice(len(vals), n_show, replace=False)
                x = np.random.normal(
                    loc=pos_map[(b, s)],
                    scale=0.02,
                    size=n_show,
                )

                ax.scatter(
                    x,
                    vals[idx],
                    alpha=0.2,
                    s=8,
                    color=color_map[s],
                    edgecolors="none",
                )

    handles = [
        Patch(
            facecolor=color_map["Real"],
            edgecolor=color_map["Real"],
            alpha=0.5,
            label="Real",
        ),
        Patch(
            facecolor=color_map["Predicted"],
            edgecolor=color_map["Predicted"],
            alpha=0.5,
            label="Predicted",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.1, 1.05),
        frameon=False,
        fontsize=7,
        handlelength=1.2,
    )

    ax.set_xlim(0.65, 3.35)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_xlabel("Expression bin")
    ax.set_ylabel("Strength")
    ax.set_title("Real vs Predictor on E. coli test set", pad=6)

    lims = get_full_limits(plot_df["value"], pad_ratio=0.06, min_pad=0.2)
    if lims:
        ax.set_ylim(*lims)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_scatter(eval_df: pd.DataFrame, metrics: dict, out_png: str, out_pdf: str):
    fig, ax = plt.subplots(figsize=(4.3, 4.0))

    color_map = {
        "low": "#1f77b4",
        "mid": "#f0ad4e",
        "high": "#d62728",
    }

    for label in LABEL_ORDER:
        sub = eval_df[eval_df["label"] == label].copy()

        if len(sub) == 0:
            continue

        n_show = min(len(sub), 1500)

        if len(sub) > n_show:
            sub = sub.sample(n=n_show, random_state=42)

        ax.scatter(
            sub["strength"],
            sub["pred_strength"],
            s=10,
            alpha=0.35,
            color=color_map[label],
            edgecolors="none",
            label=label,
        )

    all_vals = pd.concat(
        [eval_df["strength"], eval_df["pred_strength"]],
        ignore_index=True,
    )

    lims = get_full_limits(all_vals, pad_ratio=0.05, min_pad=0.2)

    if lims:
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.plot(
            lims,
            lims,
            linestyle="--",
            linewidth=1.0,
            color="gray",
        )

    txt = (
        f"Pearson r = {metrics['pearson_r']:.3f}\n"
        f"Spearman ρ = {metrics['spearman_rho']:.3f}\n"
        f"MSE = {metrics['mse']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"n = {metrics['n']}"
    )

    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    ax.set_xlabel("Real strength")
    ax.set_ylabel("Predicted strength")
    ax.set_title("Predictor performance on E. coli test set", pad=6)

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=7,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ================== 主函数 ==================
def main():
    real_df = load_real_df(REAL_CSV)

    eval_df = attach_predictions(real_df)

    metrics = compute_metrics(eval_df)
    metrics_by_label = compute_metrics_by_label(eval_df)

    print("\n=== Predictor metrics on real sequences ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n=== Predictor metrics by label ===")
    print(metrics_by_label)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)

    metrics_by_label.to_csv(
        os.path.join(OUT_DIR, "predictor_metrics_by_label.csv"),
        index=False,
    )

    plot_df = build_strength_plot_df(eval_df)

    summary_df = summarize(plot_df)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    plot_strength_box(plot_df, OUT_BOXPLOT, OUT_BOXPLOT_PDF)
    plot_scatter(eval_df, metrics, OUT_SCATTER, OUT_SCATTER_PDF)

    print(f"\n预测结果表已保存: {OUT_PRED_CSV}")
    print(f"summary 已保存: {OUT_SUMMARY_CSV}")
    print(f"metrics 已保存: {OUT_METRICS_CSV}")
    print(f"metrics by label 已保存: {os.path.join(OUT_DIR, 'predictor_metrics_by_label.csv')}")
    print(f"箱线图 PNG 已保存: {OUT_BOXPLOT}")
    print(f"箱线图 PDF 已保存: {OUT_BOXPLOT_PDF}")
    print(f"散点图 PNG 已保存: {OUT_SCATTER}")
    print(f"散点图 PDF 已保存: {OUT_SCATTER_PDF}")


if __name__ == "__main__":
    main()