"""
ism_step1_compute_batch.py — FIXED v3 (Yeast SC 100-Sample Batch Version)
改动：
  1. 从单条代表序列升级为对每个条件随机抽取 N_SAMPLE=100 条序列
  2. 提取 Oracle 加载至循环外部，避免重复加载降低效率
  3. 保存文件名保持兼容，矩阵 shape 扩展为 (N, 80, 4) 供 mean logo 脚本使用
  4. 保留原始 ISM 矩阵，不进行零中心化
"""

import os, sys, gc, warnings, logging, random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ─────────────────────────────────────────
# ★ 路径配置
# ─────────────────────────────────────────
GEN_LOW_TXT       = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/low.txt"
GEN_MID_TXT       = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/mid.txt"
GEN_HIGH_TXT      = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/high.txt"
ORACLE_DIR        = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_CONDITIONS = "defined_media"
NPY_OUT_DIR       = "/home/yt/Code/DNA-Diffusion/lunwen/ism_npy_SC"
os.makedirs(NPY_OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# ★ 关键参数
# ─────────────────────────────────────────
N_SAMPLE    = 100      # 每个条件抽取的序列数
RAND_SEED   = 42
FULL_LEN    = 110
CORE_START  = 17
CORE_LEN    = 80
LEFT_FLANK  = "TGCATTTTTTTCACATC"
RIGHT_FLANK = "GGTTACGGCTGTT"

NT_ORDER    = ["A", "C", "G", "T"]
NT2IDX      = {nt: i for i, nt in enumerate(NT_ORDER)}
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
            if not line or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if len(seq) >= CORE_LEN:
                seqs.append(seq[:CORE_LEN])
    print(f"  Read {len(seqs)} seqs from {os.path.basename(path)}")
    return seqs

def seq_to_onehot_110(seq_80):
    full = LEFT_FLANK + seq_80 + RIGHT_FLANK
    full = full[:FULL_LEN] if len(full) >= FULL_LEN else "N"*(FULL_LEN-len(full)) + full
    mat = np.zeros((FULL_LEN, 4), dtype=np.float32)
    for i, nt in enumerate(full):
        if nt in NT2IDX:
            mat[i, NT2IDX[nt]] = 1.0
    return mat

def seq_to_onehot_core(seq_80):
    mat = np.zeros((CORE_LEN, 4), dtype=np.float32)
    for i, nt in enumerate(seq_80[:CORE_LEN]):
        if nt in NT2IDX:
            mat[i, NT2IDX[nt]] = 1.0
    return mat

# ─────────────────────────────────────────
# Oracle 加载
# ─────────────────────────────────────────
def load_oracle_clean():
    import tensorflow as tf
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    from aux import load_model

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()
    gc.collect()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    model, scaler, batch_size = load_model(ORACLE_CONDITIONS)
    x_ph    = tf.compat.v1.placeholder(tf.float32, shape=(1, FULL_LEN, 4))
    pred_op = tf.squeeze(model(x_ph, training=False))

    uninit = sess.run(tf.compat.v1.report_uninitialized_variables())
    if len(uninit) > 0:
        sess.run(tf.compat.v1.global_variables_initializer())
        model, scaler, batch_size = load_model(ORACLE_CONDITIONS)
        pred_op = tf.squeeze(model(x_ph, training=False))

    print(f"  Model variable mean: {np.mean(sess.run(model.trainable_variables[0])):.6f}")
    return sess, x_ph, pred_op, scaler

def predict_physical(oh_110, sess, x_ph, pred_op, scaler):
    raw = float(sess.run(pred_op, feed_dict={x_ph: oh_110[np.newaxis]}))
    return float(scaler.inverse_transform(np.array([[raw]]))[0, 0])

# ─────────────────────────────────────────
# 批量 ISM 核心逻辑：对样本组跑完整突变
# ─────────────────────────────────────────
def compute_ism_batch(seqs, sess, x_ph, pred_op, scaler):
    """
    对给定的多条序列做完整 ISM
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
        oh_ref    = seq_to_onehot_110(seq)
        ref_score = predict_physical(oh_ref, sess, x_ph, pred_op, scaler)

        mut_matrix   = np.zeros((CORE_LEN, 4), dtype=np.float32)
        one_hot_core = seq_to_onehot_core(seq)

        for pos in range(CORE_LEN):
            real_pos = pos + CORE_START
            for nt_idx in range(4):
                oh_mut = oh_ref.copy()
                oh_mut[real_pos] = 0.0
                oh_mut[real_pos, nt_idx] = 1.0
                mut_score = predict_physical(oh_mut, sess, x_ph, pred_op, scaler)
                mut_matrix[pos, nt_idx] = ref_score - mut_score

        hyp_batch[n] = mut_matrix
        oh_batch[n]  = one_hot_core
        scores[n]    = ref_score

        # 进度条
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
    print(f"STEP 1 (batch): ISM for Yeast SC, N_SAMPLE={N_SAMPLE} per condition")

    # 固定随机种子
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    seqs_dict = {
        "low":  read_txt_sequences(GEN_LOW_TXT),
        "mid":  read_txt_sequences(GEN_MID_TXT),
        "high": read_txt_sequences(GEN_HIGH_TXT),
    }

    print("\nLoading Oracle Predictor (once for all)...")
    sess, x_ph, pred_op, scaler = load_oracle_clean()

    for label in LABEL_ORDER:
        print(f"\n{'─'*60}")
        print(f"[{label.upper()}]")

        all_seqs = seqs_dict[label]
        n_take   = min(N_SAMPLE, len(all_seqs))
        sampled  = random.sample(all_seqs, n_take)
        print(f"  Sampled {n_take} / {len(all_seqs)} sequences")
        print(f"  Total forward passes: {n_take * CORE_LEN * 4:,}")

        # 批量 ISM 计算
        print(f"  Running batch ISM...")
        hyp_batch, oh_batch, scores = compute_ism_batch(
            sampled, sess, x_ph, pred_op, scaler
        )

        # 保存为 (N, 80, 4) 矩阵结构
        prefix = os.path.join(NPY_OUT_DIR, label)
        np.save(f"{prefix}_hyp.npy",    hyp_batch)   # (N, 80, 4)
        np.save(f"{prefix}_oh.npy",     oh_batch)    # (N, 80, 4)
        np.save(f"{prefix}_scores.npy", scores)      # (N,)
        
        with open(f"{prefix}_seqs.txt", "w") as f:
            for s in sampled:
                f.write(s + "\n")

        print(f"  Saved: shape={hyp_batch.shape}")
        print(f"  Score stats: min={scores.min():.4f}"
              f"  median={np.median(scores):.4f}"
              f"  max={scores.max():.4f}")

    # 关闭 Tensorflow 会话
    sess.close()
    gc.collect()

    print(f"\n{'='*60}")
    print(f"STEP 1 done. NPY saved to: {NPY_OUT_DIR}")
    print(f"Now run plot_mean_ism_logo.py to generate the Yeast figure.")

if __name__ == "__main__":
    main()