"""
plot_importance_logo_ISM.py
============================
用 ISM（In Silico Mutagenesis）复现论文 Figure 2E 风格
"""

import os, sys, gc, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ─────────────────────────────────────────
# ★ 路径配置
# ─────────────────────────────────────────
GEN_LOW_TXT       = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/low.txt"
GEN_MID_TXT       = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/mid.txt"
GEN_HIGH_TXT      = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/high.txt"
ORACLE_DIR        = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_CONDITIONS = "defined_media"
JASPAR_MEME       = "/home/yt/Code/DNA-Diffusion/lunwen/JASPAR2026_CORE_fungi_non-redundant_pfms_meme.txt"
OUT_DIR           = "/home/yt/Code/DNA-Diffusion/lunwen/importance_logo_SC_Av"
os.makedirs(OUT_DIR, exist_ok=True)

LEFT_FLANK  = "TGCATTTTTTTCACATC"   # 17bp
RIGHT_FLANK = "GGTTACGGCTGTT"       # 13bp
FULL_LEN    = 110
CORE_START  = 17
CORE_END    = 97
CORE_LEN    = 80
NT_ORDER    = ["A", "C", "G", "T"]
NT2IDX      = {nt: i for i, nt in enumerate(NT_ORDER)}
LABEL_ORDER = ["low", "mid", "high"]
N_SEQS      = 20    # ★ ISM 较慢，每条序列需 80×3=240 次前向，建议先用20
TOP_MOTIFS  = 3

NATURE_RC = {
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 9,
    "axes.titleweight": "bold", "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
}

# ─────────────────────────────────────────
# 序列 IO
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
            if seq:
                seqs.append(seq)
    print(f"  Read {len(seqs)} seqs from {os.path.basename(path)}")
    return seqs

def seq_to_onehot_110(seq_80):
    full_seq = LEFT_FLANK + seq_80 + RIGHT_FLANK
    if len(full_seq) > FULL_LEN:
        full_seq = full_seq[-FULL_LEN:]
    elif len(full_seq) < FULL_LEN:
        full_seq = "N" * (FULL_LEN - len(full_seq)) + full_seq
    mat = np.zeros((FULL_LEN, 4), dtype=np.float32)
    for i, nt in enumerate(full_seq):
        if nt in NT2IDX:
            mat[i, NT2IDX[nt]] = 1.0
    return mat

# ─────────────────────────────────────────
# 加载 Oracle 模型
# ─────────────────────────────────────────
def load_oracle():
    import tensorflow as tf
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    from aux import load_model
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()
    graph = tf.Graph()
    with graph.as_default():
        model, scaler, batch_size = load_model(ORACLE_CONDITIONS)
    print(f"  Model input shape: {model.input_shape}")
    return model, scaler, graph

# ─────────────────────────────────────────
# ★ ISM：只需前向预测，建一次图
# ─────────────────────────────────────────
def build_pred_graph(model, graph):
    """建预测图，返回 (sess, x_ph, pred_op)"""
    import tensorflow as tf
    with graph.as_default():
        x_ph    = tf.placeholder(tf.float32, shape=(1, FULL_LEN, 4), name="x_input")
        pred    = model(x_ph, training=False)
        pred_op = tf.squeeze(pred)   # 标量预测值

        sess = tf.Session(graph=graph)
        uninit = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit.append(var)
        if uninit:
            sess.run(tf.variables_initializer(uninit))
    return sess, x_ph, pred_op


def compute_ism_one(seq_80, sess, x_ph, pred_op):
    """
    返回 (80, 4)：每个位置只有参考碱基列有值，其余为0
    值 = 该位置突变为其他3种碱基的平均预测压降（>0 表示该碱基重要）
    """
    oh_ref    = seq_to_onehot_110(seq_80)
    ref_score = sess.run(pred_op, feed_dict={x_ph: oh_ref[np.newaxis]})[()]

    importance = np.zeros((CORE_LEN, 4), dtype=np.float32)

    for pos in range(CORE_LEN):
        real_pos = pos + CORE_START
        orig_idx = int(np.argmax(oh_ref[real_pos]))

        drops = []
        for nt_idx in range(4):
            if nt_idx == orig_idx:
                continue
            oh_mut = oh_ref.copy()
            oh_mut[real_pos]         = 0.0
            oh_mut[real_pos, nt_idx] = 1.0
            mut_score = sess.run(pred_op,
                                 feed_dict={x_ph: oh_mut[np.newaxis]})[()]
            drops.append(ref_score - mut_score)

        # ★ 只在参考碱基列赋值，其余列=0
        mean_drop = float(np.mean(drops))
        importance[pos, orig_idx] = max(mean_drop, 0.0)  # 截断负值

    return importance


def compute_mean_ism(seqs, sess, x_ph, pred_op, n_max=20):
    seqs  = seqs[:n_max]
    accum = np.zeros((CORE_LEN, 4), dtype=np.float64)
    for i, seq in enumerate(seqs):
        accum += compute_ism_one(seq, sess, x_ph, pred_op)
        print(f"    [{i+1}/{len(seqs)}] ISM done")
    return (accum / len(seqs)).astype(np.float32)

# ─────────────────────────────────────────
# JASPAR motif 扫描（不变）
# ─────────────────────────────────────────
def parse_jaspar_meme(path):
    motifs = []
    cur_name, cur_rows, in_matrix = None, [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if cur_name and cur_rows:
                    motifs.append({"name": cur_name,
                                   "matrix": np.array(cur_rows, dtype=np.float32)})
                parts    = line.split()
                cur_name = parts[2] if len(parts) > 2 else parts[1]
                cur_rows = []; in_matrix = False
            elif line.startswith("letter-probability matrix"):
                in_matrix = True
            elif in_matrix and line and line[0].isdigit():
                vals = list(map(float, line.split()))
                if len(vals) == 4:
                    cur_rows.append(vals)
            elif in_matrix and (line == "" or line.startswith("URL") or
                                line.startswith("MOTIF")):
                in_matrix = False
    if cur_name and cur_rows:
        motifs.append({"name": cur_name,
                       "matrix": np.array(cur_rows, dtype=np.float32)})
    print(f"  Parsed {len(motifs)} motifs.")
    return motifs


def scan_motif(importance_mat, motif_mat):
    L, k = len(importance_mat), len(motif_mat)
    if k > L:
        return 0, k, -np.inf
    # 只用正值部分（原始碱基贡献）扫描
    pos_imp = np.clip(importance_mat, 0, None)
    scores  = [np.sum(pos_imp[s:s+k] * motif_mat) for s in range(L - k + 1)]
    best    = int(np.argmax(scores))
    return best, best + k, scores[best]


def find_top_motifs(importance_mat, jaspar_motifs, top_n=3):
    hits = []
    for motif in jaspar_motifs:
        start, end, score = scan_motif(importance_mat, motif["matrix"])
        hits.append((score, motif["name"], start, end))
    hits.sort(key=lambda x: -x[0])
    selected, used = [], set()
    for score, name, start, end in hits:
        pos = set(range(start, end))
        if not pos & used:
            selected.append((name, start, end))
            used |= pos
        if len(selected) >= top_n:
            break
    return selected

# ─────────────────────────────────────────
# ★ 绘图：论文 Figure 2E 风格
# ─────────────────────────────────────────
def plot_logo(label, importance_mat, top_motifs, out_prefix, n_seqs):
    with plt.rc_context(NATURE_RC):
        # ★ 论文图2E比例：宽长窄，约7:1
        fig, ax = plt.subplots(figsize=(10, 1.8), constrained_layout=True)

        df = pd.DataFrame(
            importance_mat,
            columns=NT_ORDER,
            index=np.arange(CORE_LEN)
        )

        # ★ 与论文配色一致：A绿 C蓝 G橙 T红
        COLOR_SCHEME = {"A": "#00A651", "C": "#0070C0", "G": "#F7941D", "T": "#FF0000"}

        logo = logomaker.Logo(
            df, ax=ax,
            color_scheme=COLOR_SCHEME,
            vpad=0.05,
            width=0.9,
            flip_below=False,    # ★ 只显示正值，不翻转
            show_spines=False,
        )
        logo.style_xticks(anchor=0, spacing=10, fmt="%d")

        ax.set_ylim(bottom=0)

        # ★ 虚线框标注 motif
        for name, start, end in top_motifs:
            y_max = ax.get_ylim()[1]
            rect = plt.Rectangle(
                (start - 0.5, 0), end - start, y_max * 0.97,
                linewidth=1.0, edgecolor="#B8860B",
                facecolor="none", linestyle="--", zorder=3
            )
            ax.add_patch(rect)
            ax.text(
                (start + end) / 2 - 0.5, y_max * 0.97,
                name, ha="center", va="top",
                fontsize=6, color="#333333", fontweight="bold"
            )

        # ★ 坐标轴标签与论文完全一致
        ax.set_xlabel("Position", fontsize=7)
        ax.set_ylabel("Importance\nScore", fontsize=7)
        ax.set_xlim(-0.5, CORE_LEN - 0.5)
        ax.tick_params(axis="both", labelsize=6)

        # ★ 无标题（论文Figure 2E无标题，面板标签在图外）
        # 如需标题取消注释：
        # ax.set_title(f"{label.capitalize()} (n={n_seqs})", fontsize=7)

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_prefix}.pdf / .png")
# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
def main():
    print("Loading sequences...")
    seqs_dict = {
        "low":  read_txt_sequences(GEN_LOW_TXT),
        "mid":  read_txt_sequences(GEN_MID_TXT),
        "high": read_txt_sequences(GEN_HIGH_TXT),
    }

    print("\nLoading oracle model...")
    model, scaler, graph = load_oracle()

    # ★ 建预测图（ISM只需前向）
    print("\nBuilding prediction graph...")
    sess, x_ph, pred_op = build_pred_graph(model, graph)

    print("\nParsing JASPAR motifs...")
    jaspar_motifs = parse_jaspar_meme(JASPAR_MEME)

    for label in LABEL_ORDER:
        print(f"\n{'='*50}\nProcessing: {label.upper()}")
        seqs = seqs_dict[label].copy()
        np.random.shuffle(seqs)
        n = min(N_SEQS, len(seqs))

        print(f"  ISM over {n} sequences ({n * CORE_LEN * 3} forward passes)...")
        imp = compute_mean_ism(seqs, sess, x_ph, pred_op, n_max=n)

        np.save(os.path.join(OUT_DIR, f"ism_{label}.npy"), imp)

        print("  Scanning JASPAR motifs...")
        top_motifs = find_top_motifs(imp, jaspar_motifs, top_n=TOP_MOTIFS)
        print(f"  Top motifs: {top_motifs}")

        out_prefix = os.path.join(OUT_DIR, f"ism_logo_{label}")
        plot_logo(label, imp, top_motifs, out_prefix, n_seqs=n)

    sess.close()
    print(f"\nAll done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()