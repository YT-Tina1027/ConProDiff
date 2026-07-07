import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

# ─────────────────────────────────────────
# ★ 路径配置
# ─────────────────────────────────────────
NPY_OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/ism_npy_ecoli"
JASPAR_MEME  = "/home/yt/Code/DNA-Diffusion/lunwen/motif_databases/ECOLI/SwissRegulon_e_coli.meme"
PLOT_OUT_DIR = "/home/yt/Code/DNA-Diffusion/lunwen/importance_logo_modisco_ecoli1"
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 常量
# ─────────────────────────────────────────
CORE_LEN             = 165
NT_ORDER             = ["A", "C", "G", "T"]
LABEL_ORDER          = ["low", "mid", "high"]
TOP_MOTIFS           = 3
MATCH_CORR_THRESHOLD = 0.58
SIGNAL_STD_RATIO     = 0.6

COLOR_SCHEME = {"A": "#107C41", "C": "#1F4E79", "G": "#F19020", "T": "#C00000"}

FIG_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "axes.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
}

# ═══════════════════════════════════════════
# 1. 数据读取
# ═══════════════════════════════════════════
def load_ism_results(label):
    prefix = os.path.join(NPY_OUT_DIR, label)
    hyp    = np.load(f"{prefix}_hyp.npy")[0]
    oh     = np.load(f"{prefix}_oh.npy")[0]
    scores = np.load(f"{prefix}_scores.npy")
    return hyp, oh, scores

def load_ism_results_by_idx(label, idx):
    """按指定索引加载单条序列的 ISM 结果"""
    prefix = os.path.join(NPY_OUT_DIR, label)
    hyp    = np.load(f"{prefix}_hyp.npy")[idx]
    oh     = np.load(f"{prefix}_oh.npy")[idx]
    scores = np.load(f"{prefix}_scores.npy")
    return hyp, oh, scores

def pick_representative_idx(scores, label):
    """
    high -> 最大值索引
    low  -> 最小值索引
    mid  -> 最接近中位数的索引
    """
    if label == "high":
        idx = int(np.argmax(scores))
    elif label == "low":
        idx = int(np.argmin(scores))
    else:  # mid
        median = np.median(scores)
        idx = int(np.argmin(np.abs(scores - median)))
    return idx

def hyp_to_display(hyp_contrib, one_hot_core):
    display = np.zeros((CORE_LEN, 4), dtype=np.float32)
    for pos in range(CORE_LEN):
        ref_nt   = int(np.argmax(one_hot_core[pos]))
        other_nt = [i for i in range(4) if i != ref_nt]
        display[pos, ref_nt] = float(np.mean([hyp_contrib[pos, i] for i in other_nt]))
    return display

# ═══════════════════════════════════════════
# 2. JASPAR 解析与数学计算
# ═══════════════════════════════════════════
def parse_tf_name(acc):
    return re.sub(r'_\d+(-\d+)?$', '', acc) or acc

def parse_jaspar_meme(path):
    motifs = []
    cur_name, cur_rows, in_matrix = None, [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                if cur_name and cur_rows:
                    mat = np.array(cur_rows, dtype=np.float32)
                    rs  = mat.sum(axis=1, keepdims=True)
                    mat = mat / np.where(rs == 0, 1, rs)
                    motifs.append({"name": cur_name, "matrix": mat})
                parts    = line.split()
                raw_name = parts[2] if len(parts) > 2 else parts[1]
                cur_name = parse_tf_name(raw_name)
                cur_rows = []
                in_matrix = False
            elif line.startswith("letter-probability matrix"):
                in_matrix = True
            elif in_matrix and line and line[0].isdigit():
                vals = list(map(float, line.split()))
                if len(vals) == 4:
                    cur_rows.append(vals)
    if cur_name and cur_rows:
        mat = np.array(cur_rows, dtype=np.float32)
        rs  = mat.sum(axis=1, keepdims=True)
        mat = mat / np.where(rs == 0, 1, rs)
        motifs.append({"name": cur_name, "matrix": mat})
    print(f"  Parsed {len(motifs)} motifs from JASPAR.")
    return motifs

def rc_pwm(m):
    return m[::-1][:, [3, 2, 1, 0]]

def information_content(pwm):
    eps = 1e-9
    return np.clip(2.0 + np.sum(pwm * np.log2(pwm + eps), axis=1), 0, 2)

def ism_segment_to_query_pwm(hyp_contrib, start, end):
    seg    = hyp_contrib[start:end].copy()
    prefer = -seg
    prefer -= prefer.max(axis=1, keepdims=True)
    exp_p  = np.exp(prefer)
    return (exp_p / exp_p.sum(axis=1, keepdims=True)).astype(np.float32)

def ic_weighted_pearson(query_pwm, ref_pwm):
    k, m = len(query_pwm), len(ref_pwm)
    if k > m:
        return -1.0
    best     = -1.0
    query_ic = information_content(query_pwm)
    for strand in [query_pwm, rc_pwm(query_pwm)]:
        qf = (strand * query_ic[:, None]).flatten()
        if qf.std() < 1e-6:
            continue
        for s in range(m - k + 1):
            seg    = ref_pwm[s:s+k]
            ref_ic = information_content(seg)
            sf     = (seg * ref_ic[:, None]).flatten()
            if sf.std() < 1e-6:
                continue
            corr = float(np.corrcoef(qf, sf)[0, 1])
            if corr > best:
                best = corr
    return best

# ═══════════════════════════════════════════
# 3. Motif 检测
# ═══════════════════════════════════════════
def detect_and_match_motifs_scientific(hyp_contrib, one_hot_core, jaspar_motifs):
    centered = hyp_contrib - hyp_contrib.mean(axis=1, keepdims=True)
    raw_sig  = np.array([
        centered[pos, int(np.argmax(one_hot_core[pos]))]
        for pos in range(CORE_LEN)
    ], dtype=np.float32)

    abs_sig      = np.abs(raw_sig)
    noise_cutoff = float(np.mean(abs_sig) + SIGNAL_STD_RATIO * np.std(abs_sig))

    all_evaluated_segments = []

    for direction in ["positive", "negative"]:
        valid_positions = (raw_sig >= noise_cutoff) if direction == "positive" \
                          else (raw_sig <= -noise_cutoff)

        for w_size in range(5, 20):
            for start in range(CORE_LEN - w_size + 1):
                end = start + w_size
                if np.mean(valid_positions[start:end]) < 0.70:
                    continue
                energy    = float(np.sum(abs_sig[start:end]))
                query_pwm = ism_segment_to_query_pwm(hyp_contrib, start, end)
                best_name, best_corr = None, -1.0
                for motif in jaspar_motifs:
                    corr = ic_weighted_pearson(query_pwm, motif["matrix"])
                    if corr > best_corr:
                        best_corr = corr
                        best_name = motif["name"]
                if best_corr >= MATCH_CORR_THRESHOLD:
                    all_evaluated_segments.append(
                        (best_corr, w_size, energy, start, end, best_name, direction)
                    )

    all_evaluated_segments.sort(key=lambda x: (-x[1], -x[0], -x[2]))

    final_hits, occupied_positions = [], set()
    for corr, w_size, eng, s, e, tf_name, direction in all_evaluated_segments:
        span = set(range(s, e))
        if not (span & occupied_positions):
            final_hits.append((tf_name, s, e, direction))
            occupied_positions |= span
        if len(final_hits) >= TOP_MOTIFS:
            break

    final_hits.sort(key=lambda x: x[1])
    return final_hits

# ═══════════════════════════════════════════
# 4. 绘图
# ═══════════════════════════════════════════
def plot_logo(label, importance_mat, top_motifs, out_prefix, pred_score):
    with plt.rc_context(FIG_RC):
        fig, ax = plt.subplots(figsize=(10.0, 1.2))
        df = pd.DataFrame(importance_mat, columns=NT_ORDER, index=np.arange(CORE_LEN))

        logo = logomaker.Logo(
            df, ax=ax, color_scheme=COLOR_SCHEME,
            vpad=0.01, width=0.85, flip_below=True, show_spines=False
        )

        ymin, ymax = importance_mat.min(), importance_mat.max()
        margin     = max(abs(ymax), abs(ymin)) * 0.30
        y_lim_min  = (ymin - margin) if ymin < -1e-4 else -abs(ymax) * 0.15
        y_lim_max  = (ymax + margin) if ymax >  1e-4 else  abs(ymin) * 0.15

        if abs(y_lim_max - y_lim_min) < 1e-6:
            y_lim_min, y_lim_max = -0.01, 0.01
        ax.set_ylim(y_lim_min, y_lim_max)

        yticks = [0.0]
        if ymax > 0.2: yticks.append(round(ymax, 3))
        if ymin < -0.2: yticks.append(round(ymin, 3))
        ax.set_yticks(sorted(yticks))
        logo.style_xticks(anchor=0, spacing=20, fmt="%d")
        ax.axhline(0, color="#444444", linewidth=0.5, zorder=2)

        ax.text(0.99, 0.05, f"Oracle expression: {pred_score:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.5, color="#888888",
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.5))

        y_range = y_lim_max - y_lim_min

        for tf_name, start, end, direction in top_motifs:
            if direction == "negative":
                box_bot = y_lim_min
                box_top = 0.0
                text_y  = box_bot + (y_range * 0.01)
                text_va = "bottom"
            else:
                box_bot = 0.0
                box_top = y_lim_max
                text_y  = box_top - (y_range * 0.01)
                text_va = "top"

            rect = plt.Rectangle(
                (start - 0.4, box_bot), end - start - 0.2, box_top - box_bot,
                linewidth=0.6, edgecolor="#B8860B", facecolor="none",
                linestyle="--", zorder=3
            )
            ax.add_patch(rect)
            ax.text((start + end) / 2 - 0.5, text_y, tf_name,
                    ha="center", va=text_va, fontsize=5.0,
                    color="black", zorder=4, clip_on=True,
                    bbox=dict(boxstyle="square,pad=0.05", facecolor="white",
                              edgecolor="none", alpha=0.6))

        ax.set_ylabel("Importance Score", fontsize=6.5, labelpad=2)
        ax.set_xlim(-0.5, CORE_LEN - 0.5)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

# ═══════════════════════════════════════════
# 5. 主程序入口
# ═══════════════════════════════════════════
def main():
    print("=" * 60)
    print("STEP 2: Smart Tick-Cleaner Visualizer Initializing...")
    jaspar_motifs = parse_jaspar_meme(JASPAR_MEME)

    for label in LABEL_ORDER:
        print(f"\n[{label.upper()}]")

        # 读取全部 scores，按策略挑选代表序列
        scores_all = np.load(os.path.join(NPY_OUT_DIR, f"{label}_scores.npy"))
        idx        = pick_representative_idx(scores_all, label)
        pred_score = float(scores_all[idx])
        print(f"  Selected idx={idx}, score={pred_score:.4f}  "
              f"({'max' if label == 'high' else 'min' if label == 'low' else 'median'})")

        hyp, oh, _ = load_ism_results_by_idx(label, idx)
        importance_display = hyp_to_display(hyp, oh)

        top_motifs = detect_and_match_motifs_scientific(importance_display, oh, jaspar_motifs)

        print(f"  Verified True TF Matches:")
        for name, s, e, direction in top_motifs:
            print(f"    pos {s:2d}–{e:2d} ({e-s:2d}bp) [{direction[:3].upper()}] -> {name}")

        out_prefix = os.path.join(PLOT_OUT_DIR, f"ism_logo_clean_{label}")
        plot_logo(label, importance_display, top_motifs, out_prefix, pred_score)

if __name__ == "__main__":
    main()