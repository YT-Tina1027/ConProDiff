"""
ism_step1_compute_ecoli_batch.py
对每个条件随机抽取 N_SAMPLE 条序列，各跑一次 ISM，
保存为 (N, 165, 4)，供 mean logo 脚本使用
"""

import os, sys, gc, warnings, random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# ★ 路径配置
# ─────────────────────────────────────────
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/high.txt"

REAL_CSV             = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
PREDICTOR_MODEL_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"
BIO_ROOT             = "/home/yt/Code/Bio_diffusion_gan"

NPY_OUT_DIR = "/home/yt/Code/DNA-Diffusion/lunwen/ism_npy_ecoli"
os.makedirs(NPY_OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# ★ 关键参数
# ─────────────────────────────────────────
N_SAMPLE  = 100     # 每个条件抽取的序列数，可改为 100（计算量翻倍）
RAND_SEED = 42
CORE_LEN  = 165

NT_ORDER = ["A", "C", "G", "T"]
NT2IDX   = {nt: i for i, nt in enumerate(NT_ORDER)}
LABEL_ORDER = ["low", "mid", "high"]

# ─────────────────────────────────────────
# 序列处理
# ─────────────────────────────────────────
def standardize_seq(s):
    s = str(s).upper().strip().replace("U", "T")
    return "".join(c for c in s if c in "ATCG")

def read_txt_sequences(path):
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"): continue
            seq = standardize_seq(line)
            if len(seq) >= CORE_LEN:
                seqs.append(seq[:CORE_LEN])
    print(f"  Read {len(seqs)} seqs from {os.path.basename(path)}")
    return seqs

def seq_to_onehot(seq):
    mat = np.zeros((CORE_LEN, 4), dtype=np.float32)
    for i, nt in enumerate(seq[:CORE_LEN]):
        if nt in NT2IDX:
            mat[i, NT2IDX[nt]] = 1.0
    return mat

# ─────────────────────────────────────────
# 加载 LSTM 预测器
# ─────────────────────────────────────────
def load_predictor():
    if BIO_ROOT not in sys.path:
        sys.path.insert(0, BIO_ROOT)
    from Predictor.predictor_models import LSTMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_df = pd.read_csv(REAL_CSV)
    real_df.columns = [c.strip().lower() for c in real_df.columns]
    real_df["strength"] = pd.to_numeric(real_df["strength"], errors="coerce")
    real_df = real_df.dropna(subset=["strength"])

    scaler = StandardScaler()
    scaler.fit(real_df["strength"].values.reshape(-1, 1))

    model = LSTMModel(4, 256, 1, 0.2, 0.001)
    state = torch.load(PREDICTOR_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"  LSTM predictor loaded on {device}")
    return model, scaler, device

def predict_single(oh_mat, model, scaler, device):
    x = torch.tensor(oh_mat[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_norm = model(x).detach().cpu().numpy().reshape(-1, 1)
    return float(scaler.inverse_transform(y_norm)[0, 0])

# ─────────────────────────────────────────
# 批量 ISM：对 N 条序列各跑一次
# ─────────────────────────────────────────
def compute_ism_batch(seqs, model, scaler, device, label):
    """
    对 seqs 中每条序列做完整 ISM
    返回:
      hyp_batch : (N, CORE_LEN, 4)
      oh_batch  : (N, CORE_LEN, 4)
      scores    : (N,)
    """
    N = len(seqs)
    hyp_batch = np.zeros((N, CORE_LEN, 4), dtype=np.float32)
    oh_batch  = np.zeros((N, CORE_LEN, 4), dtype=np.float32)
    scores    = np.zeros(N, dtype=np.float32)

    for n, seq in enumerate(seqs):
        oh_ref    = seq_to_onehot(seq)
        ref_score = predict_single(oh_ref, model, scaler, device)

        mut_matrix = np.zeros((CORE_LEN, 4), dtype=np.float32)
        for pos in range(CORE_LEN):
            for nt_idx in range(4):
                oh_mut = oh_ref.copy()
                oh_mut[pos]         = 0.0
                oh_mut[pos, nt_idx] = 1.0
                mut_score = predict_single(oh_mut, model, scaler, device)
                mut_matrix[pos, nt_idx] = ref_score - mut_score

        hyp_batch[n] = mut_matrix
        oh_batch[n]  = oh_ref
        scores[n]    = ref_score

        # 进度
        passes_done = (n + 1) * CORE_LEN * 4
        passes_total = N * CORE_LEN * 4
        print(f"    seq {n+1:3d}/{N}  score={ref_score:.4f}"
              f"  ({passes_done}/{passes_total} forward passes)", end="\r")

    print()  # 换行
    return hyp_batch, oh_batch, scores

# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"STEP 1 (batch): ISM for E. coli, N_SAMPLE={N_SAMPLE} per condition")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    seqs_dict = {
        "low":  read_txt_sequences(GEN_LOW_TXT),
        "mid":  read_txt_sequences(GEN_MID_TXT),
        "high": read_txt_sequences(GEN_HIGH_TXT),
    }

    print("\nLoading LSTM predictor...")
    model, scaler, device = load_predictor()

    for label in LABEL_ORDER:
        print(f"\n{'─'*60}")
        print(f"[{label.upper()}]")

        all_seqs = seqs_dict[label]
        n_take   = min(N_SAMPLE, len(all_seqs))
        sampled  = random.sample(all_seqs, n_take)
        print(f"  Sampled {n_take} / {len(all_seqs)} sequences")
        print(f"  Total forward passes: {n_take * CORE_LEN * 4:,}")

        hyp_batch, oh_batch, scores = compute_ism_batch(
            sampled, model, scaler, device, label
        )

        prefix = os.path.join(NPY_OUT_DIR, label)
        np.save(f"{prefix}_hyp.npy",    hyp_batch)   # (N, 165, 4)
        np.save(f"{prefix}_oh.npy",     oh_batch)    # (N, 165, 4)
        np.save(f"{prefix}_scores.npy", scores)      # (N,)

        print(f"  Saved: shape={hyp_batch.shape}")
        print(f"  Score stats: min={scores.min():.4f}"
              f"  median={np.median(scores):.4f}"
              f"  max={scores.max():.4f}")

    print(f"\n{'='*60}")
    print(f"Done. NPY saved to: {NPY_OUT_DIR}")
    print(f"Now run plot_mean_ism_logo.py to generate the figure.")

if __name__ == "__main__":
    main()