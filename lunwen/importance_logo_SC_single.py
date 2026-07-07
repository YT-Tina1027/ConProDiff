"""
plot_importance_logo_modisco.py
================================
严格按论文 Figure 2E 方法复现：

论文流程：
  1. 对多条生成序列计算 ISM hypothetical contributions (N × L × 4)
  2. 将全部序列的 ISM 结果输入 TF-MoDISco，聚类发现 de novo motif patterns
  3. 将 TF-MoDISco patterns 与 JASPAR PPM 做相关性匹配，给 pattern 命名
  4. 找出 top-1 代表序列中各 pattern 出现的 seqlet 位置
  5. 在 top-1 序列的 ISM importance logo 上高亮标注 motif 位置

关键区别（与旧版JASPAR滑窗方法的对比）：
  - 旧版：单条序列 × JASPAR 滑窗点积 → 找最优位置
  - 新版：多条序列 ISM → TF-MoDISco 聚类 → JASPAR 匹配 → 在代表序列上定位

依赖：
  pip install modisco-lite logomaker matplotlib pandas numpy tensorflow
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
# ★ 路径配置（与原脚本一致）
# ─────────────────────────────────────────
GEN_LOW_TXT       = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/low.txt"
GEN_MID_TXT       = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/mid.txt"
GEN_HIGH_TXT      = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc3.0/high.txt"
ORACLE_DIR        = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_CONDITIONS = "defined_media"
JASPAR_MEME       = "/home/yt/Code/DNA-Diffusion/lunwen/JASPAR2026_CORE_fungi_non-redundant_pfms_meme.txt"
OUT_DIR           = "/home/yt/Code/DNA-Diffusion/lunwen/importance_logo_modisco"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 常量
# ─────────────────────────────────────────
LEFT_FLANK  = "TGCATTTTTTTCACATC"   # 17 bp
RIGHT_FLANK = "GGTTACGGCTGTT"       # 13 bp
FULL_LEN    = 110
CORE_START  = 17
CORE_END    = 97
CORE_LEN    = 80
NT_ORDER    = ["A", "C", "G", "T"]
NT2IDX      = {nt: i for i, nt in enumerate(NT_ORDER)}
LABEL_ORDER = ["low", "mid", "high"]

# ── ISM / TF-MoDISco 参数 ──
SCORE_TOPK          = 300   # 先批量预测，取前 300 候选
ISM_N               = 100   # 对 top-100 跑完整 ISM，喂给 TF-MoDISco
                             # 每条 80 × 4 = 320 次前向传播，共 32,000 次
MODISCO_MIN_SEQLETS = 10    # pattern 至少需要这么多 seqlets
TOP_MOTIFS          = 3     # 在图上标注的最多 motif 数
JASPAR_CORR_THRESH  = 0.65  # pattern 与 JASPAR motif 相关系数阈值

# ─────────────────────────────────────────
# rc 参数（与原脚本一致）
# ─────────────────────────────────────────
FIG_RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          7,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,
}

COLOR_SCHEME = {"A": "#00A651", "C": "#0070C0", "G": "#F7941D", "T": "#FF0000"}

# ═══════════════════════════════════════════
# 1. 序列 IO
# ═══════════════════════════════════════════

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
    """80bp 核心序列 → 加 flank → one-hot (110, 4)"""
    full_seq = LEFT_FLANK + seq_80 + RIGHT_FLANK
    full_seq = full_seq[-FULL_LEN:] if len(full_seq) > FULL_LEN \
               else "N" * (FULL_LEN - len(full_seq)) + full_seq
    mat = np.zeros((FULL_LEN, 4), dtype=np.float32)
    for i, nt in enumerate(full_seq):
        if nt in NT2IDX:
            mat[i, NT2IDX[nt]] = 1.0
    return mat

def seq_to_onehot_core(seq_80):
    """80bp 核心序列 → one-hot (80, 4)，用于 TF-MoDISco 输入"""
    mat = np.zeros((CORE_LEN, 4), dtype=np.float32)
    for i, nt in enumerate(seq_80[:CORE_LEN]):
        if nt in NT2IDX:
            mat[i, NT2IDX[nt]] = 1.0
    return mat

# ═══════════════════════════════════════════
# 2. 加载 Oracle 模型
# ═══════════════════════════════════════════

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

def build_pred_graph(model, graph):
    """返回 (sess, x_ph, pred_op)"""
    import tensorflow as tf
    with graph.as_default():
        x_ph    = tf.placeholder(tf.float32, shape=(1, FULL_LEN, 4), name="x_input")
        pred    = model(x_ph, training=False)
        pred_op = tf.squeeze(pred)
        sess    = tf.Session(graph=graph)
        uninit  = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit.append(var)
        if uninit:
            sess.run(tf.variables_initializer(uninit))
    return sess, x_ph, pred_op

def predict_single(oh_110, sess, x_ph, pred_op):
    return float(sess.run(pred_op, feed_dict={x_ph: oh_110[np.newaxis]}))

# ═══════════════════════════════════════════
# 3. 批量打分，选 top 序列
# ═══════════════════════════════════════════

def select_top_seqs(seqs, sess, x_ph, pred_op, score_topk, ism_n):
    """
    1. 对前 score_topk 条批量预测
    2. 按预测分排序，返回前 ism_n 条 (score, seq) 列表
    返回列表已按分数降序排列，索引 0 = 最优代表序列
    """
    pool = seqs[:score_topk]
    print(f"  Scoring {len(pool)} sequences...")
    scored = []
    for seq in pool:
        oh = seq_to_onehot_110(seq)
        s  = predict_single(oh, sess, x_ph, pred_op)
        scored.append((s, seq))
    scored.sort(key=lambda x: -x[0])
    top = scored[:ism_n]
    print(f"  Top-{ism_n} score range: "
          f"{top[-1][0]:.4f} ~ {top[0][0]:.4f}")
    return top   # list of (score, seq_80)

# ═══════════════════════════════════════════
# 4. ★ ISM 核心（论文方法）
#    输出完整 (L, 4) hypothetical contributions
#    供 TF-MoDISco 使用
# ═══════════════════════════════════════════

def compute_hyp_ism(seq_80, sess, x_ph, pred_op):
    """
    论文 ISM 方法：对每个位置的每种碱基计算预测分，
    用均值中心化得到 hypothetical contribution。

    公式：
        mut_scores[p, n]   = score( seq 中 position p 替换为 nucleotide n )
        mean_score[p]      = mean over all 4 nt at position p
        hyp_contrib[p, n]  = mut_scores[p, n] - mean_score[p]

    返回：
        hyp_contrib  (80, 4) float32   ← 喂给 TF-MoDISco
        one_hot_core (80, 4) float32   ← 实际序列的 one-hot
        ref_score    float             ← 原始序列预测分（用于展示）
    """
    oh_ref    = seq_to_onehot_110(seq_80)
    ref_score = predict_single(oh_ref, sess, x_ph, pred_op)

    hyp_contrib  = np.zeros((CORE_LEN, 4), dtype=np.float32)
    one_hot_core = seq_to_onehot_core(seq_80)

    for pos in range(CORE_LEN):
        real_pos     = pos + CORE_START
        mut_scores   = np.zeros(4, dtype=np.float32)

        for nt_idx in range(4):
            oh_mut                     = oh_ref.copy()
            oh_mut[real_pos]           = 0.0
            oh_mut[real_pos, nt_idx]   = 1.0
            mut_scores[nt_idx]         = predict_single(oh_mut, sess, x_ph, pred_op)

        # 均值中心化（标准 ISM hypothetical contribution）
        hyp_contrib[pos] = mut_scores - mut_scores.mean()

    return hyp_contrib, one_hot_core, ref_score

def compute_ism_batch(top_seqs, sess, x_ph, pred_op):
    """
    对 top_seqs（列表 [(score, seq_80), ...]）批量计算 ISM。
    返回：
        hyp_batch    (N, 80, 4)  hypothetical contributions
        oh_batch     (N, 80, 4)  one-hot core sequences
        ref_scores   (N,)        各序列预测分
    索引 0 = top-1 代表序列
    """
    N = len(top_seqs)
    hyp_batch  = np.zeros((N, CORE_LEN, 4), dtype=np.float32)
    oh_batch   = np.zeros((N, CORE_LEN, 4), dtype=np.float32)
    ref_scores = np.zeros(N, dtype=np.float32)

    for i, (score, seq) in enumerate(top_seqs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  ISM progress: {i+1}/{N}")
        hyp, oh, rs       = compute_hyp_ism(seq, sess, x_ph, pred_op)
        hyp_batch[i]      = hyp
        oh_batch[i]       = oh
        ref_scores[i]     = rs

    print(f"  ISM done. {N} sequences × {CORE_LEN} positions × 4 nt")
    return hyp_batch, oh_batch, ref_scores

# ═══════════════════════════════════════════
# 5. ★ TF-MoDISco（论文方法）
# ═══════════════════════════════════════════

def run_tfmodisco(hyp_batch, oh_batch, min_seqlets=MODISCO_MIN_SEQLETS):
    """
    调用 modisco-lite 对全部序列的 ISM importance 做 pattern 聚类。

    输入：
        hyp_batch  (N, L, 4)  ISM hypothetical contributions
        oh_batch   (N, L, 4)  one-hot sequences
    返回：
        pos_patterns  list[Pattern]  活化 patterns（actual contrib > 0）
        neg_patterns  list[Pattern]  抑制 patterns
    """
    try:
        import modiscolite
    except ImportError:
        raise ImportError(
            "请先安装 modisco-lite：pip install modisco-lite"
        )

    print("  Running TF-MoDISco...")
    print(f"    Input shape: hyp={hyp_batch.shape}, oh={oh_batch.shape}")

    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs = hyp_batch,   # (N, L, 4)
        one_hot                = oh_batch,    # (N, L, 4)
        max_seqlets_per_metacluster = 20000,
        sliding_window_size    = 15,
        flank_size             = 5,
        target_seqlet_fdr      = 0.05,
        n_leiden_runs          = 2,
        trim_to_window_size    = 15,
        initial_flank_to_add   = 3,
        verbose                = False,
    )

    # 过滤 seqlet 数不足的 pattern
    pos_patterns = [p for p in pos_patterns
                    if len(p.seqlets) >= min_seqlets]
    neg_patterns = [p for p in neg_patterns
                    if len(p.seqlets) >= min_seqlets]

    print(f"  TF-MoDISco done. "
          f"pos patterns: {len(pos_patterns)}, "
          f"neg patterns: {len(neg_patterns)}")
    return pos_patterns, neg_patterns

# ═══════════════════════════════════════════
# 6. 将 TF-MoDISco patterns 匹配到 JASPAR
# ═══════════════════════════════════════════

def parse_jaspar_meme(path):
    """解析 JASPAR MEME 文件，返回 [{name, matrix(L×4)}]，列顺序 A C G T"""
    motifs = []
    cur_name, cur_rows, in_matrix = None, [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if cur_name and cur_rows:
                    motifs.append({"name":   cur_name,
                                   "matrix": np.array(cur_rows, dtype=np.float32)})
                parts     = line.split()
                cur_name  = parts[2] if len(parts) > 2 else parts[1]
                cur_rows  = []
                in_matrix = False
            elif line.startswith("letter-probability matrix"):
                in_matrix = True
            elif in_matrix and line and line[0].isdigit():
                vals = list(map(float, line.split()))
                if len(vals) == 4:
                    cur_rows.append(vals)
            elif in_matrix and (line == "" or line.startswith("URL")
                                or line.startswith("MOTIF")):
                in_matrix = False
    if cur_name and cur_rows:
        motifs.append({"name":   cur_name,
                       "matrix": np.array(cur_rows, dtype=np.float32)})
    print(f"  Parsed {len(motifs)} JASPAR motifs.")
    return motifs

def get_pattern_pwm(pattern):
    """
    从 TF-MoDISco pattern 提取归一化的 PPM (L, 4)。
    使用 pattern 的 sequence_attributions（各位置各碱基的平均贡献）。
    """
    # modisco-lite 的 pattern 对象有 contrib_scores 属性
    try:
        # 尝试访问 pattern 的 per-position 碱基频率
        pwm = pattern.sequence_attributions  # shape (L, 4)
    except AttributeError:
        try:
            pwm = pattern.contrib_scores      # 备用属性名
        except AttributeError:
            # 如果都不存在，从 seqlets 手动计算碱基频率
            seqlet_ohs = []
            for seqlet in pattern.seqlets:
                seqlet_ohs.append(seqlet.sequence)   # (k, 4)
            if not seqlet_ohs:
                return None
            pwm = np.mean(seqlet_ohs, axis=0)        # (k, 4)

    # 归一化为 PPM（每行概率和 = 1）
    pwm = np.array(pwm, dtype=np.float32)
    row_sums = pwm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return pwm / row_sums

def pearson_corr_slide(pwm_query, pwm_ref):
    """
    滑窗皮尔逊相关：在 pwm_ref 上寻找与 pwm_query 最相似的子窗口。
    同时比较正向和反向互补（RC）。
    返回最高相关系数。
    """
    def rc_pwm(m):
        """反向互补：逆序行，交换 A↔T (col0↔col3)，C↔G (col1↔col2)"""
        m = m[::-1].copy()
        m = m[:, [3, 2, 1, 0]]
        return m

    k = len(pwm_query)
    m = len(pwm_ref)
    if k > m:
        return -1.0

    best = -1.0
    q_flat = pwm_query.flatten()
    if q_flat.std() < 1e-6:
        return best

    for strand_pwm in [pwm_query, rc_pwm(pwm_query)]:
        s_flat = strand_pwm.flatten()
        for start in range(m - k + 1):
            sub   = pwm_ref[start:start + k].flatten()
            if sub.std() < 1e-6:
                continue
            corr  = float(np.corrcoef(s_flat, sub)[0, 1])
            if corr > best:
                best = corr
    return best

def match_patterns_to_jaspar(patterns, jaspar_motifs,
                              threshold=JASPAR_CORR_THRESH):
    """
    将 TF-MoDISco patterns 匹配到最相似的 JASPAR motif。
    返回：list of (pattern_idx, tf_name, corr_score)
    仅保留相关系数 >= threshold 的匹配。
    """
    results = []
    for pi, pattern in enumerate(patterns):
        pwm = get_pattern_pwm(pattern)
        if pwm is None or len(pwm) == 0:
            continue

        best_name, best_corr = None, -1.0
        for motif in jaspar_motifs:
            ref_pwm = motif["matrix"]
            corr    = pearson_corr_slide(pwm, ref_pwm)
            if corr > best_corr:
                best_corr = corr
                best_name = motif["name"]

        if best_corr >= threshold:
            results.append((pi, best_name, best_corr))
            print(f"    Pattern {pi}: {best_name} (corr={best_corr:.3f}, "
                  f"seqlets={len(pattern.seqlets)})")
        else:
            print(f"    Pattern {pi}: no match (best_corr={best_corr:.3f})")

    return results

# ═══════════════════════════════════════════
# 7. 在代表序列中定位 motif 的 seqlet 位置
# ═══════════════════════════════════════════

def find_motifs_in_representative(pos_patterns, matched_names,
                                   rep_seq_idx=0, top_n=TOP_MOTIFS):
    """
    在 TF-MoDISco 的 seqlets 里找属于代表序列（rep_seq_idx=0）的实例，
    提取 motif 在核心序列上的起止位置。

    pos_patterns  : TF-MoDISco 返回的 pattern 列表
    matched_names : [(pattern_idx, tf_name, corr), ...]
    rep_seq_idx   : 代表序列在 ISM batch 中的索引（通常 = 0）
    top_n         : 最多返回几个 motif

    返回：list of (tf_name, start, end)，已按起始位置排序，贪心去重
    """
    hits = []  # (start, end, tf_name)

    for pi, tf_name, corr in matched_names:
        pattern = pos_patterns[pi]
        for seqlet in pattern.seqlets:
            # seqlet.example_idx : 该 seqlet 属于哪条序列
            if seqlet.example_idx != rep_seq_idx:
                continue
            start = int(seqlet.start)
            end   = int(seqlet.end)
            # 确保范围在 core 内
            start = max(0, min(start, CORE_LEN - 1))
            end   = max(start + 1, min(end, CORE_LEN))
            hits.append((start, end, tf_name, corr))

    if not hits:
        print("  ⚠ 代表序列中未找到 seqlet，"
              "尝试使用全体 seqlets 中最高相关的位置...")
        # fallback：取每个 pattern 得分最高的 seqlet（不限序列）
        for pi, tf_name, corr in matched_names:
            pattern = pos_patterns[pi]
            if not pattern.seqlets:
                continue
            seqlet  = pattern.seqlets[0]   # 已按贡献排序
            start   = max(0, int(seqlet.start))
            end     = min(CORE_LEN, int(seqlet.end))
            hits.append((start, end, tf_name, corr))

    # 贪心去重：按相关系数排序，不允许位置重叠
    hits.sort(key=lambda x: -x[3])
    selected, used_pos = [], set()
    for start, end, tf_name, _ in hits:
        pos_set = set(range(start, end))
        if not pos_set & used_pos:
            selected.append((tf_name, start, end))
            used_pos |= pos_set
        if len(selected) >= top_n:
            break

    selected.sort(key=lambda x: x[1])  # 按位置排序，方便画图
    return selected

# ═══════════════════════════════════════════
# 8. 从 hypothetical contribution 提取展示用 importance
# ═══════════════════════════════════════════

def hyp_to_display_importance(hyp_contrib, one_hot_core):
    """
    Figure 2E 的显示量 = 实际碱基在 hypothetical contribution 中的值。
      display[p, ref_nt]  = max( hyp_contrib[p, ref_nt], 0 )
      display[p, other]   = 0

    即：只显示参考碱基，高度 = 该碱基的"均值中心化 ISM 分"，截断负值。
    这与论文描述一致：
      "Height represents the per-nucleotide importance score."
    """
    display = np.zeros_like(hyp_contrib)
    for pos in range(CORE_LEN):
        ref_nt = int(np.argmax(one_hot_core[pos]))
        val    = hyp_contrib[pos, ref_nt]
        display[pos, ref_nt] = max(float(val), 0.0)
    return display

# ═══════════════════════════════════════════
# 9. 绘图（Figure 2E 风格）
# ═══════════════════════════════════════════

def plot_logo(label, importance_mat, top_motifs, out_prefix, pred_score):
    """
    importance_mat : (80, 4)，只有参考碱基列有正值
    top_motifs     : [(tf_name, start, end), ...]
    """
    with plt.rc_context(FIG_RC):
        fig, ax = plt.subplots(figsize=(8, 1.4), constrained_layout=True)

        df = pd.DataFrame(importance_mat,
                          columns=NT_ORDER,
                          index=np.arange(CORE_LEN))

        logo = logomaker.Logo(
            df, ax=ax,
            color_scheme=COLOR_SCHEME,
            vpad=0.05,
            width=0.9,
            flip_below=False,
            show_spines=False,
        )
        logo.style_xticks(anchor=0, spacing=20, fmt="%d")
        ax.set_ylim(bottom=0)

        # 虚线框标注 motif
        y_max = ax.get_ylim()[1]
        for tf_name, start, end in top_motifs:
            rect = plt.Rectangle(
                (start - 0.5, 0),
                end - start,
                y_max * 0.97,
                linewidth=0.8,
                edgecolor="#B8860B",
                facecolor="none",
                linestyle="--",
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                (start + end) / 2 - 0.5,
                y_max * 0.96,
                tf_name,
                ha="center", va="top",
                fontsize=5.5,
                color="#222222",
                fontweight="bold",
            )

        ax.set_xlabel("Position", fontsize=7)
        ax.set_ylabel("Importance\nScore", fontsize=7, labelpad=2)
        ax.set_xlim(-0.5, CORE_LEN - 0.5)
        ax.tick_params(axis="both", labelsize=6)

        y_max = ax.get_ylim()[1]
        ax.set_yticks([0, round(y_max, 2)])

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_prefix}.pdf / .png  "
              f"(pred_score={pred_score:.4f}, "
              f"motifs={[m[0] for m in top_motifs]})")

# ═══════════════════════════════════════════
# 10. 主流程
# ═══════════════════════════════════════════

def main():
    # ── 读取序列 ──────────────────────────────
    print("=" * 55)
    print("Step 1: Loading sequences")
    seqs_dict = {
        "low":  read_txt_sequences(GEN_LOW_TXT),
        "mid":  read_txt_sequences(GEN_MID_TXT),
        "high": read_txt_sequences(GEN_HIGH_TXT),
    }

    # ── 加载模型 ──────────────────────────────
    print("\nStep 2: Loading oracle model")
    model, scaler, graph = load_oracle()

    print("\nStep 3: Building TF1 prediction graph")
    sess, x_ph, pred_op = build_pred_graph(model, graph)

    # ── 解析 JASPAR ───────────────────────────
    print("\nStep 4: Parsing JASPAR motif database")
    jaspar_motifs = parse_jaspar_meme(JASPAR_MEME)

    # ── 每组处理 ──────────────────────────────
    for label in LABEL_ORDER:
        print(f"\n{'='*55}")
        print(f"Processing: {label.upper()}")

        seqs = seqs_dict[label].copy()
        np.random.shuffle(seqs)

        # Step A：批量打分，选 top-ISM_N 序列
        print(f"\n  [A] Scoring top-{SCORE_TOPK}, selecting top-{ISM_N}")
        top_seqs = select_top_seqs(
            seqs, sess, x_ph, pred_op,
            score_topk=SCORE_TOPK,
            ism_n=ISM_N,
        )
        # top_seqs[0] 是预测分最高的代表序列

        # Step B：对全部 top-ISM_N 条序列跑 ISM
        print(f"\n  [B] Computing ISM for {ISM_N} sequences "
              f"({ISM_N * CORE_LEN * 4} forward passes)...")
        hyp_batch, oh_batch, ref_scores = compute_ism_batch(
            top_seqs, sess, x_ph, pred_op
        )

        # 保存 npy 备用
        np.save(os.path.join(OUT_DIR, f"hyp_{label}.npy"), hyp_batch)
        np.save(os.path.join(OUT_DIR, f"oh_{label}.npy"),  oh_batch)

        # Step C：TF-MoDISco（论文核心步骤）
        print(f"\n  [C] Running TF-MoDISco on {ISM_N} sequences...")
        pos_patterns, neg_patterns = run_tfmodisco(hyp_batch, oh_batch)

        if not pos_patterns:
            print("  ⚠ TF-MoDISco 未找到正向 patterns，跳过该组")
            continue

        # Step D：将 patterns 匹配到 JASPAR TF 名称
        print(f"\n  [D] Matching {len(pos_patterns)} patterns to JASPAR...")
        matched = match_patterns_to_jaspar(pos_patterns, jaspar_motifs)

        if not matched:
            print("  ⚠ 未能匹配任何 JASPAR motif，跳过该组")
            continue

        # Step E：在代表序列（index=0）中定位 seqlet 位置
        print(f"\n  [E] Locating motifs in representative sequence...")
        top_motifs = find_motifs_in_representative(
            pos_patterns, matched,
            rep_seq_idx=0,
            top_n=TOP_MOTIFS,
        )
        print(f"  Motifs found: {top_motifs}")

        # Step F：计算代表序列的展示 importance（Figure 2E 风格）
        imp_display = hyp_to_display_importance(hyp_batch[0], oh_batch[0])

        # Step G：画图
        print(f"\n  [G] Plotting...")
        out_prefix = os.path.join(OUT_DIR, f"ism_logo_{label}")
        plot_logo(
            label, imp_display, top_motifs, out_prefix,
            pred_score=float(ref_scores[0]),
        )

    sess.close()
    print(f"\n{'='*55}")
    print(f"All done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()