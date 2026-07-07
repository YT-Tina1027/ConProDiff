import os
import sys
import gc
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== 静音设置 ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ================== 全局绘图风格 ==================
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False
})

# ================== 路径配置 ==================
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/high.txt"

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/defined_SC_Ura_core80_812.csv"

ORACLE_DIR = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_MODEL_CONDITIONS = "defined_media"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/analysis_condition_hit_tertile1"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PRED_CSV = os.path.join(OUT_DIR, "generated_with_pred_and_bins.csv")
OUT_COUNT_CSV = os.path.join(OUT_DIR, "condition_hit_count_matrix.csv")
OUT_PROP_CSV = os.path.join(OUT_DIR, "condition_hit_proportion_matrix.csv")
OUT_HEATMAP_PNG = os.path.join(OUT_DIR, "condition_hit_heatmap.png")
OUT_HEATMAP_PDF = os.path.join(OUT_DIR, "condition_hit_heatmap.pdf")
OUT_THRESHOLD_CSV = os.path.join(OUT_DIR, "tertile_thresholds.csv")

# ================== 参数设置 ==================
# 使用三分位阈值：
# low  : pred_strength <= Q1
# mid  : Q1 < pred_strength <= Q2
# high : pred_strength > Q2
USE_TERTILE_THRESHOLD = True

LABEL_ORDER = ["low", "mid", "high"]

SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]


# ================== 基础函数 ==================
def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ATCG"])


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


def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")


# ================== 数据加载 ==================
def load_generated_txts(low_txt, mid_txt, high_txt):
    rows = []

    for seq in read_txt_sequences(low_txt):
        rows.append({"target_bin": "low", "sequence_std": seq})

    for seq in read_txt_sequences(mid_txt):
        rows.append({"target_bin": "mid", "sequence_std": seq})

    for seq in read_txt_sequences(high_txt):
        rows.append({"target_bin": "high", "sequence_std": seq})

    return pd.DataFrame(rows)


def infer_tertile_thresholds_from_real_csv(real_csv):
    """
    从真实数据 strength 中计算三分位阈值。

    分箱规则：
    low  : strength <= Q1
    mid  : Q1 < strength <= Q2
    high : strength > Q2
    """
    if not os.path.exists(real_csv):
        raise FileNotFoundError(f"未找到真实数据文件: {real_csv}")

    df = pd.read_csv(real_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    if "strength" not in df.columns:
        raise ValueError("真实 CSV 里必须有 strength 列")

    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    df = df.dropna(subset=["strength"]).copy()

    q1 = float(df["strength"].quantile(1.0 / 3.0))
    q2 = float(df["strength"].quantile(2.0 / 3.0))

    return q1, q2


# ================== Oracle 预测 ==================
def predict_sequences_with_oracle(sequences, oracle_dir, model_conditions):
    import tensorflow as tf

    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)

    from aux import load_model, evaluate_model

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    fitness_function_graph = tf.Graph()
    with fitness_function_graph.as_default():
        model, scaler, batch_size = load_model(model_conditions)

    preds = evaluate_model(sequences, model, scaler, batch_size, fitness_function_graph)
    return np.asarray(preds, dtype=float).reshape(-1)


def attach_predictions(gen_df, oracle_dir, model_conditions):
    seqs = gen_df["sequence_std"].tolist()
    preds = predict_sequences_with_oracle(seqs, oracle_dir, model_conditions)

    out = gen_df.copy()
    out["pred_strength"] = pd.to_numeric(preds, errors="coerce")
    out = out.dropna(subset=["pred_strength"]).reset_index(drop=True)

    return out


# ================== 分箱 ==================
def assign_bin(x, low_mid_threshold, mid_high_threshold):
    """
    和三分位论文表述保持一致：

    low  : x <= Q1
    mid  : Q1 < x <= Q2
    high : x > Q2
    """
    if x <= low_mid_threshold:
        return "low"
    elif x <= mid_high_threshold:
        return "mid"
    else:
        return "high"


def add_predicted_bins(df, low_mid_threshold, mid_high_threshold):
    out = df.copy()
    out["pred_bin"] = out["pred_strength"].apply(
        lambda x: assign_bin(x, low_mid_threshold, mid_high_threshold)
    )
    return out


# ================== 统计矩阵 ==================
def build_count_and_prop_matrix(df):
    count_mat = pd.crosstab(
        pd.Categorical(df["target_bin"], categories=LABEL_ORDER, ordered=True),
        pd.Categorical(df["pred_bin"], categories=LABEL_ORDER, ordered=True),
        dropna=False
    )

    count_mat.index.name = "target_bin"
    count_mat.columns.name = "pred_bin"

    prop_mat = count_mat.div(count_mat.sum(axis=1).replace(0, np.nan), axis=0)

    return count_mat, prop_mat


def compute_hit_summary(prop_mat):
    rows = []
    for label in LABEL_ORDER:
        rows.append({
            "target_bin": label,
            "hit_rate": float(prop_mat.loc[label, label])
        })
    return pd.DataFrame(rows)


# ================== 绘图 ==================
def plot_condition_hit_heatmap(prop_mat, count_mat, out_png, out_pdf, q1, q2):
    mat = prop_mat.loc[LABEL_ORDER, LABEL_ORDER].values

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_yticklabels(LABEL_ORDER)

    ax.set_xlabel("Predicted bin by oracle")
    ax.set_ylabel("Target condition")
    ax.set_title("Condition hit rate", pad=8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            prop_val = mat[i, j]
            count_val = count_mat.iloc[i, j]
            text = f"{prop_val:.2f}\n(n={count_val})"
            txt_color = "white" if prop_val > 0.5 else "black"
            ax.text(
                j, i, text,
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion")

    # 在图下方标注三分位阈值，方便论文解释
    threshold_text = (
        f"Tertile thresholds: "
        f"low ≤ {q1:.3f}, "
        f"{q1:.3f} < mid ≤ {q2:.3f}, "
        f"high > {q2:.3f}"
    )

    fig.text(
        0.5, -0.02,
        threshold_text,
        ha="center",
        va="top",
        fontsize=7
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# ================== 主函数 ==================
def main():
    print("\n==============================")
    print("Loading generated sequences")
    print("==============================")

    gen_df = load_generated_txts(GEN_LOW_TXT, GEN_MID_TXT, GEN_HIGH_TXT)

    print("\nGenerated sequence counts:")
    print(gen_df.groupby("target_bin").size())

    print("\n==============================")
    print("Oracle prediction")
    print("==============================")

    gen_df = attach_predictions(gen_df, ORACLE_DIR, ORACLE_MODEL_CONDITIONS)

    print("\n==============================")
    print("Computing tertile thresholds")
    print("==============================")

    if USE_TERTILE_THRESHOLD:
        low_mid_threshold, mid_high_threshold = infer_tertile_thresholds_from_real_csv(REAL_CSV)
        print(
            f"Using tertile thresholds from REAL_CSV: "
            f"Q1={low_mid_threshold:.6f}, Q2={mid_high_threshold:.6f}"
        )
    else:
        raise ValueError("当前版本建议使用 USE_TERTILE_THRESHOLD=True。")

    threshold_df = pd.DataFrame([{
        "strategy": "tertile",
        "low_rule": f"pred_strength <= {low_mid_threshold:.6f}",
        "mid_rule": f"{low_mid_threshold:.6f} < pred_strength <= {mid_high_threshold:.6f}",
        "high_rule": f"pred_strength > {mid_high_threshold:.6f}",
        "q1_low_mid_threshold": low_mid_threshold,
        "q2_mid_high_threshold": mid_high_threshold,
    }])
    threshold_df.to_csv(OUT_THRESHOLD_CSV, index=False)

    print("\nBinning rules:")
    print(f"low  : pred_strength <= {low_mid_threshold:.6f}")
    print(f"mid  : {low_mid_threshold:.6f} < pred_strength <= {mid_high_threshold:.6f}")
    print(f"high : pred_strength > {mid_high_threshold:.6f}")

    print("\n==============================")
    print("Building hit matrix")
    print("==============================")

    gen_df = add_predicted_bins(gen_df, low_mid_threshold, mid_high_threshold)
    gen_df.to_csv(OUT_PRED_CSV, index=False)

    count_mat, prop_mat = build_count_and_prop_matrix(gen_df)
    count_mat.to_csv(OUT_COUNT_CSV)
    prop_mat.to_csv(OUT_PROP_CSV)

    hit_summary = compute_hit_summary(prop_mat)
    hit_summary.to_csv(os.path.join(OUT_DIR, "condition_hit_summary.csv"), index=False)

    print("\nCount matrix:")
    print(count_mat)

    print("\nProportion matrix:")
    print(prop_mat)

    print("\nPer-condition hit rate:")
    print(hit_summary)

    print("\n==============================")
    print("Plotting heatmap")
    print("==============================")

    plot_condition_hit_heatmap(
        prop_mat,
        count_mat,
        OUT_HEATMAP_PNG,
        OUT_HEATMAP_PDF,
        low_mid_threshold,
        mid_high_threshold
    )

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Saved predicted CSV: {OUT_PRED_CSV}")
    print(f"Saved count matrix: {OUT_COUNT_CSV}")
    print(f"Saved proportion matrix: {OUT_PROP_CSV}")
    print(f"Saved threshold CSV: {OUT_THRESHOLD_CSV}")
    print(f"Saved heatmap PNG: {OUT_HEATMAP_PNG}")
    print(f"Saved heatmap PDF: {OUT_HEATMAP_PDF}")


if __name__ == "__main__":
    main()