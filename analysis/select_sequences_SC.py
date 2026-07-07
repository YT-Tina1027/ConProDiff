"""
select_sequences_SC_simple.py
==============================
S. cerevisiae 启动子序列筛选 —— 精简版

核心逻辑（去除所有生物学硬过滤，只保留 pred 分层 + 多样性）：

  1. Oracle 批量预测所有生成序列的 pred
  2. 按 pred 分层筛选：
       high → pred 最高的 top N（贪心多样性）
       mid  → pred 最接近中心区间的 top N（贪心多样性）
       low  → pred 最低的 top N（贪心多样性）
  3. 天然对照（nature，来自 REAL_CSV）：
       各组 pred 上限 = 对应生成组 pred 最高值（不优于生成序列）
  4. 随机阴性对照：pred 严格低于生成 low 组最低值（尽量）
  5. 最终保证：random ≤ generated_low ≤ generated_mid ≤ generated_high
                nature_xxx ≤ generated_xxx（各组独立上限）
"""

import os, sys, gc as _gc, re, warnings, logging
import numpy as np
import pandas as pd
from itertools import combinations

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# =========================================================
# ★ 配置区（只改这里）★
# =========================================================

GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/high.txt"

REAL_CSV          = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
ORACLE_DIR        = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_CONDITIONS = "defined_media"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/result/experiments_SC/selected_sequences"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN        = 80     # S. cerevisiae 核心启动子长度
N_SELECT       = 5      # 每组生成序列数量
N_NATURE       = 2      # 每组天然对照数量（0 = 不选天然对照）
N_RANDOM       = 2      # 随机阴性对照数量
N_RANDOM_POOL  = 2000   # 随机候选池大小（酵母 Oracle 基线较高，需大池）
GREEDY_POOL    = 80     # 排序后送入贪心的最大候选数
SEED           = 42

# =========================================================
# 工具函数
# =========================================================

def standardize_seq(s):
    s = str(s).upper().strip().replace("U", "T")
    return "".join(ch for ch in s if ch in "ATCG")

def read_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件: {path}")
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            s = standardize_seq(line)
            if s:
                seqs.append(s)
    assert seqs, f"文件中无有效序列: {path}"
    return seqs

def gc_content(seq):
    seq = standardize_seq(seq)
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")

def hamming(s1, s2):
    if len(s1) == len(s2):
        return sum(a != b for a, b in zip(s1, s2))
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = min(prev[j]+1, dp[j-1]+1,
                        prev[j-1] + (0 if s1[i-1]==s2[j-1] else 1))
    return dp[n]

def min_pairwise(seqs):
    if len(seqs) < 2:
        return 0
    return min(hamming(a, b) for a, b in combinations(seqs, 2))

def greedy_diverse(df, n):
    seqs = df["seq"].tolist()
    if len(seqs) <= n:
        return df
    selected = [0]
    while len(selected) < n:
        best_i, best_d = -1, -1
        for i in range(len(seqs)):
            if i in selected:
                continue
            d = min(hamming(seqs[i], seqs[j]) for j in selected)
            if d > best_d:
                best_d, best_i = d, i
        if best_i == -1:
            break
        selected.append(best_i)
    return df.iloc[selected].copy()

# =========================================================
# Oracle 预测
# =========================================================

def predict_sequences_oracle(sequences):
    import tensorflow as tf

    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    from aux import load_model, evaluate_model

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    _gc.collect()

    graph = tf.Graph()
    with graph.as_default():
        model, scaler, batch_size = load_model(ORACLE_CONDITIONS)

    preds = evaluate_model(sequences, model, scaler, batch_size, graph)
    return np.asarray(preds, dtype=float).reshape(-1)

# =========================================================
# 分箱边界
# =========================================================

def compute_bin_edges(real_csv):
    df = pd.read_csv(real_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next((c for c in ["strength", "activity", "score", "expression", "expr"]
                if c in df.columns), None)
    if col is None:
        raise ValueError(f"找不到表达强度列，当前列：{list(df.columns)}")
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    q33, q67 = float(np.percentile(vals, 33.333)), float(np.percentile(vals, 66.667))
    print(f"  [BinEdge] Low < {q33:.4f} ≤ Mid ≤ {q67:.4f} < High")
    return q33, q67

# =========================================================
# 生成序列筛选（核心：pred 分层 + 贪心多样性）
# =========================================================

def select_generated(seqs, preds, group, q33, q67, n=5, pool=80):
    df = pd.DataFrame({"seq": seqs, "pred": preds}).dropna()

    if group == "high":
        df = df.sort_values("pred", ascending=False)
    elif group == "low":
        df = df.sort_values("pred", ascending=True)
    else:  # mid
        center = (q33 + q67) / 2.0
        df["dist"] = (df["pred"] - center).abs()
        df = df.sort_values("dist")

    candidates = df.head(pool).reset_index(drop=True)
    result = greedy_diverse(candidates, n).copy()
    result = result.drop(columns=["dist"], errors="ignore")
    result["group"]  = group
    result["source"] = "generated"
    result["rank"]   = range(1, len(result) + 1)

    print(f"  [{group:>4}] 生成: {len(seqs)} 条 → 已选 {len(result)} 条 | "
          f"pred=[{result['pred'].min():.4f}, {result['pred'].max():.4f}] | "
          f"最小编辑距离={min_pairwise(result['seq'].tolist())}")

    return result[["rank", "group", "source", "seq", "pred"]]

# =========================================================
# 天然对照筛选
# 约束：pred 落在 [gen_pred_min, gen_pred_max] 内（不优于生成序列）
# =========================================================

def select_nature(real_csv, group, q33, q67,
                  gen_pred_min, gen_pred_max,
                  n=2, pool=80):
    df = pd.read_csv(real_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    seq_col = next((c for c in ["sequence", "seq", "dna", "promoter"]
                    if c in df.columns), None)
    str_col = next((c for c in ["strength", "activity", "score", "expression"]
                    if c in df.columns), None)
    if seq_col is None or str_col is None:
        print(f"  [nature/{group}] ⚠ 找不到序列/强度列，跳过天然对照")
        return None

    df = df[[seq_col, str_col]].copy()
    df.columns = ["seq", "pred"]
    df["seq"]  = df["seq"].apply(standardize_seq)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # 核心约束：pred 在生成序列范围内
    qualified = df[
        (df["pred"] >= gen_pred_min) &
        (df["pred"] <= gen_pred_max)
    ].copy()

    if len(qualified) < n:
        # 放宽：只保证上限（不优于生成序列最高），去掉下限
        qualified = df[df["pred"] <= gen_pred_max].copy()
        print(f"  [nature/{group}] ⚠ 严格范围内不足 {n} 条，"
              f"放宽下限（上限保持 ≤ {gen_pred_max:.4f}）")

    if len(qualified) < n:
        print(f"  [nature/{group}] ⚠ 仍不足 {n} 条，跳过天然对照")
        return None

    # 按接近生成序列 pred 中心排序
    center = (gen_pred_min + gen_pred_max) / 2.0
    qualified["dist"] = (qualified["pred"] - center).abs()
    qualified = qualified.sort_values("dist").reset_index(drop=True)

    candidates = qualified.head(pool).reset_index(drop=True)
    result = greedy_diverse(candidates, n).copy()
    result = result.drop(columns=["dist"], errors="ignore")
    result["group"]  = group
    result["source"] = "nature"
    result["rank"]   = range(1, len(result) + 1)

    print(f"  [nature/{group}] 已选 {len(result)} 条 | "
          f"pred=[{result['pred'].min():.4f}, {result['pred'].max():.4f}] "
          f"（生成范围 [{gen_pred_min:.4f}, {gen_pred_max:.4f}]）| "
          f"最小编辑距离={min_pairwise(result['seq'].tolist())}")

    return result[["rank", "group", "source", "seq", "pred"]]

# =========================================================
# 随机阴性对照（pred 严格低于 low 组最低值）
# =========================================================

def select_random(low_pred_min, n=2, pool_size=2000, length=80, seed=42):
    rng = np.random.default_rng(seed)

    for multiplier in [1, 2, 4]:
        cur_pool = pool_size * multiplier
        seqs = [
            "".join(rng.choice(list("ATCG"), size=length))
            for _ in range(cur_pool)
        ]
        print(f"  [random] 生成 {cur_pool} 条随机序列，Oracle 预测中…")
        preds = predict_sequences_oracle(seqs)
        df = pd.DataFrame({"seq": seqs, "pred": preds})
        df = df.sort_values("pred", ascending=True).reset_index(drop=True)

        strict = df[df["pred"] < low_pred_min]
        if len(strict) >= n:
            df = strict
            print(f"  [random] ✓ {len(df)} 条 pred < {low_pred_min:.4f}")
            break
        else:
            print(f"  [random] 仅 {len(strict)} 条满足 pred < {low_pred_min:.4f}，"
                  f"扩大候选池重试…")
            if multiplier == 4:
                df = df.head(max(n * 5, 10))
                print(f"  [random] ⚠ 保底：取 pred 最低的 {len(df)} 条，"
                      f"最低={df['pred'].min():.4f}（建议增大 N_RANDOM_POOL）")

    result = greedy_diverse(df, n).copy()
    result["group"]  = "random"
    result["source"] = "random"
    result["rank"]   = range(1, n + 1)

    print(f"  [random] 已选 {n} 条 | "
          f"pred=[{result['pred'].min():.4f}, {result['pred'].max():.4f}] | "
          f"最小编辑距离={min_pairwise(result['seq'].tolist())}")

    return result[["rank", "group", "source", "seq", "pred"]]

# =========================================================
# 跨组 pred 层次检查
# =========================================================

def check_pred_hierarchy(frames):
    print("\n  [pred 层次检查]")
    all_df = pd.concat(frames, ignore_index=True)
    order = ["random", "low", "mid", "high"]
    stats = {}
    for grp in order:
        sub = all_df[all_df["group"] == grp]
        if not sub.empty:
            stats[grp] = (sub["pred"].min(), sub["pred"].max())
            print(f"    {grp:>6}: pred ∈ [{stats[grp][0]:.4f}, {stats[grp][1]:.4f}]")

    prev_grp, prev_max = None, None
    for grp in order:
        if grp not in stats:
            continue
        cur_min = stats[grp][0]
        if prev_max is not None:
            gap  = cur_min - prev_max
            flag = "✓" if gap >= 0 else "⚠"
            note = "" if gap >= 0 else " ← 层次有重叠，湿实验可能难以区分"
            print(f"    {flag} {prev_grp} max → {grp} min  gap={gap:.4f}{note}")
        prev_grp, prev_max = grp, stats[grp][1]

# =========================================================
# 主函数
# =========================================================

def main():
    np.random.seed(SEED)
    print("=" * 64)
    print("  S. cerevisiae 启动子序列筛选（精简版）")
    print("  策略：pred 分层 + 贪心多样性 + 对照组不优于生成序列")
    print("=" * 64)

    # 1. 读取生成序列
    print("\n[1] 读取生成序列...")
    low_s  = read_txt(GEN_LOW_TXT)
    mid_s  = read_txt(GEN_MID_TXT)
    high_s = read_txt(GEN_HIGH_TXT)
    print(f"    low={len(low_s)}, mid={len(mid_s)}, high={len(high_s)}")

    # 2. Oracle 批量预测
    print("\n[2] Oracle 批量预测...")
    all_seqs  = low_s + mid_s + high_s
    all_preds = predict_sequences_oracle(all_seqs)
    nl, nm    = len(low_s), len(mid_s)
    low_p  = all_preds[:nl]
    mid_p  = all_preds[nl:nl+nm]
    high_p = all_preds[nl+nm:]
    print(f"    low  pred: [{low_p.min():.4f}, {low_p.max():.4f}]")
    print(f"    mid  pred: [{mid_p.min():.4f}, {mid_p.max():.4f}]")
    print(f"    high pred: [{high_p.min():.4f}, {high_p.max():.4f}]")

    # 3. 分箱边界
    print("\n[3] 计算分箱边界...")
    q33, q67 = compute_bin_edges(REAL_CSV)

    # 4. 生成序列筛选
    print("\n[4] 生成序列筛选（pred 排序 + 贪心多样性）...")
    low_sel  = select_generated(low_s,  low_p,  "low",  q33, q67, N_SELECT, GREEDY_POOL)
    mid_sel  = select_generated(mid_s,  mid_p,  "mid",  q33, q67, N_SELECT, GREEDY_POOL)
    high_sel = select_generated(high_s, high_p, "high", q33, q67, N_SELECT, GREEDY_POOL)

    # 5. 天然对照（pred 上限 = 对应生成组 pred 最高值）
    nat_parts = []
    if N_NATURE > 0:
        print("\n[5] 天然对照筛选（pred 不优于生成序列）...")
        for sel, grp in [(low_sel, "low"), (mid_sel, "mid"), (high_sel, "high")]:
            gen_min = float(sel["pred"].min())
            gen_max = float(sel["pred"].max())
            nat = select_nature(REAL_CSV, grp, q33, q67,
                                gen_min, gen_max, N_NATURE, GREEDY_POOL)
            if nat is not None:
                nat_parts.append(nat)
    else:
        print("\n[5] N_NATURE=0，跳过天然对照")

    # 6. 随机阴性对照
    print("\n[6] 随机阴性对照（pred 低于生成 low 组最低值）...")
    low_pred_min = float(low_sel["pred"].min())
    print(f"    生成 low 组最低 pred = {low_pred_min:.4f}")
    rand_df = select_random(low_pred_min, N_RANDOM, N_RANDOM_POOL, SEQ_LEN, SEED)

    # 7. 汇总
    parts = [low_sel, mid_sel, high_sel] + nat_parts + [rand_df]
    final = pd.concat(parts, ignore_index=True)
    final["gc_content"] = final["seq"].apply(gc_content)
    final["seq_len"]    = final["seq"].apply(len)

    # pred 层次检查
    check_pred_hierarchy(parts)

    # 8. 保存
    csv_path = os.path.join(OUT_DIR, "selected_SC_simple.csv")
    fa_path  = os.path.join(OUT_DIR, "selected_SC_simple.fasta")

    final.to_csv(csv_path, index=False)

    with open(fa_path, "w") as f:
        for _, row in final.iterrows():
            src, grp, rank = row["source"], row["group"].upper(), int(row["rank"])
            if src == "random":
                tag = f"R{rank}_RANDOM"
            elif src == "nature":
                tag = f"N{rank}_{grp}_NATURE"
            else:
                tag = f"S{rank}_{grp}"
            f.write(f">{tag}  pred={row['pred']:.4f}  gc={row['gc_content']:.3f}\n")
            f.write(f"{row['seq']}\n")

    # 9. 控制台汇总
    print("\n" + "=" * 64)
    print("  最终筛选结果")
    print("=" * 64)
    expected = N_SELECT * 3 + N_NATURE * 3 + N_RANDOM
    for grp in ["high", "mid", "low", "random"]:
        sub = final[final["group"] == grp]
        if sub.empty:
            continue
        d = min_pairwise(sub["seq"].tolist()) if len(sub) > 1 else "—"
        print(f"  {grp:>6}: {len(sub)} 条  "
              f"pred=[{sub['pred'].min():.4f}, {sub['pred'].max():.4f}]  "
              f"最小编辑距离={d}")
    print(f"\n  总样本数: {len(final)}（期望 {expected}）")
    print(f"\n[保存] CSV   → {csv_path}")
    print(f"[保存] FASTA → {fa_path}")
    print("\n完成！")


if __name__ == "__main__":
    main()