import os
import sys
import gc
import warnings
import logging
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/defined_SC_Ura_core80_812.csv"
ORACLE_DIR = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_MODEL_CONDITIONS = "defined_media"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/analysis_trainset_oracles"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_REAL_ORACLE_CSV = os.path.join(OUT_DIR, "real_with_defined_oracle_pred_strength_812.csv")

SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]
LABEL_ORDER = ["low", "mid", "high"]


def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join([ch for ch in s if ch in "ATCG"])


def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列为: {SEQ_COL_CANDIDATES}")


def load_real_strength_df_from_label(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到真实数据文件: {path}")

    real_df = pd.read_csv(path)
    real_df.columns = [c.strip().lower() for c in real_df.columns]

    if "strength" not in real_df.columns or "label" not in real_df.columns:
        raise ValueError("真实数据中缺少 strength 或 label 列")

    seq_col = find_seq_col(real_df)

    real_df = real_df.copy()
    real_df["strength"] = pd.to_numeric(real_df["strength"], errors="coerce")
    real_df["label"] = real_df["label"].astype(str).str.strip().str.lower()
    real_df["sequence_std"] = real_df[seq_col].astype(str).apply(standardize_seq)

    real_df = real_df.dropna(subset=["strength", "label", "sequence_std"]).reset_index(drop=True)
    real_df = real_df[real_df["sequence_std"].str.len() > 0]
    real_df = real_df[real_df["label"].isin(LABEL_ORDER)]
    real_df["group"] = real_df["label"].map(lambda x: f"real_{x}")
    return real_df.reset_index(drop=True)


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


def main():
    real_df = load_real_strength_df_from_label(REAL_CSV)
    seqs = real_df["sequence_std"].tolist()
    preds = predict_sequences_with_oracle(seqs, ORACLE_DIR, ORACLE_MODEL_CONDITIONS)

    real_df["pred_strength_oracle"] = pd.to_numeric(preds, errors="coerce")
    real_df = real_df.dropna(subset=["pred_strength_oracle"]).reset_index(drop=True)

    seq_col = "sequence" if "sequence" in real_df.columns else "sequence_std"
    real_df[[seq_col, "strength", "label", "group", "pred_strength_oracle"]].to_csv(
        OUT_REAL_ORACLE_CSV, index=False
    )
    print(f"oracle预测结果已保存: {OUT_REAL_ORACLE_CSV}")


if __name__ == "__main__":
    main()