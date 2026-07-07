"""
Step 2: 用 pymemesuite 扫描生成序列的 TF motif
参数完全按照 regLM 论文：
  - 均匀背景频率
  - 默认伪计数 0.1
  - p值阈值 0.001
  - 正反链均扫描
"""
import os, re, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pymemesuite.common import MotifFile, Sequence, Background, Alphabet, Array
from pymemesuite.fimo import FIMO

# ══════════════════════════════════════════════════════════════
# 路径（根据你的实际情况修改）
# ══════════════════════════════════════════════════════════════
JASPAR_MEME = "/home/yt/Code/DNA-Diffusion/lunwen/motif_databases/ECOLI/SwissRegulon_e_coli.meme"

# 如果是酵母序列改这里，大肠杆菌改成ecoli路径
GEN_FILES = {
    "low":  "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/low.txt",
    "mid":  "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/mid.txt",
    "high": "/home/yt/Code/DNA-Diffusion/result/experiments_ecoli/data_cfg3.0/high.txt",
}

OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/tu3_ecoli"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 真实序列路径（图3用）──────────────────────────────────────
REAL_CSV = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"
# CSV 里的列名，改成你文件实际的列名
SEQ_COL      = "sequence"
ACTIVITY_COL = "strength"

# ══════════════════════════════════════════════════════════════
# 1. 读取序列
# ══════════════════════════════════════════════════════════════
def read_seqs(path, seed=42):   # 去掉 max_n 参数
    seqs = []
    for line in open(path):
        tok = max(re.split(r"[\s,\t;|]+", line.strip()),
                  key=lambda x: len(x), default="")
        s = "".join(c for c in tok.upper() if c in "ACGT")
        if len(s) >= 20:
            seqs.append(s)
    return seqs

# ══════════════════════════════════════════════════════════════
# 2. 加载JASPAR MEME文件
#    - 读取所有motif
#    - 每个TF只保留版本号最大的（去冗余）
# ══════════════════════════════════════════════════════════════
def load_motifs(meme_file):
    all_motifs = []
    with MotifFile(meme_file) as f:
        for m in f:
            all_motifs.append(m)
    print(f"  原始motif数量: {len(all_motifs)}")

    # SwissRegulon用accession去冗余，不用name
    seen = set()
    latest_motifs = []
    for m in all_motifs:
        key = m.accession if m.accession else m.name
        if key not in seen:
            seen.add(key)
            latest_motifs.append(m)

    print(f"  去冗余后motif数量: {len(latest_motifs)}")
    return latest_motifs

# ══════════════════════════════════════════════════════════════
# 3. FIMO扫描
#    - 均匀背景频率（A=C=G=T=0.25）
#    - 伪计数 0.1（pymemesuite默认值）
#    - p值阈值 0.001
#    - 正反链均扫描（both_strands=True）
#    返回：{motif_name: 含该motif的序列比例}
# ══════════════════════════════════════════════════════════════
import re
def parse_tf_name(acc):
    """
    'H_NS_28-19'  -> 'H_NS'
    'CRP_28-1'    -> 'CRP'
    'FKH1_12-0'   -> 'FKH1'
    'Ada_28-0'    -> 'Ada'
    """
    # 去掉末尾形如 _数字-数字 或 _数字 的版本号后缀
    name = re.sub(r'_\d+(-\d+)?$', '', acc)
    return name if name else acc

def scan_sequences(seqs, motifs, pvalue_thresh=1e-3):
    seq_objs   = [Sequence(s, name=f"seq_{i}".encode())
                  for i, s in enumerate(seqs)]
    alphabet   = Alphabet.dna()
    background = Background(alphabet, Array([0.25, 0.25, 0.25, 0.25]))
    fimo       = FIMO(both_strands=True, threshold=pvalue_thresh,
                      max_stored_scores=100000)

    hits          = defaultdict(set)
    seq_motif_set = defaultdict(set)
    seq_hit_count = defaultdict(int)
    # 新增：存每个 motif 的所有命中起始位置
    motif_positions = defaultdict(list)   # {motif_name: [pos, pos, ...]}

    for m in motifs:
        acc = m.accession
        if isinstance(acc, bytes):
            acc = acc.decode()
        motif_name = parse_tf_name(acc) if acc else "unknown"

        pattern = fimo.score_motif(m, seq_objs, background)
        if pattern is None:
            continue

        me = pattern.matched_elements
        for i in range(len(me)):
            el = me[i]
            if el.pvalue is not None and el.pvalue <= pvalue_thresh:
                src = el.source.name
                if isinstance(src, bytes):
                    src = src.decode()
                idx = int(src.replace("seq_", ""))
                hits[motif_name].add(idx)
                seq_motif_set[idx].add(motif_name)
                seq_hit_count[idx] += 1
                # 存起始位置（归一化到 0~1 方便不同长度序列对齐）
                seq_len = len(seqs[idx])
                norm_pos = el.start / seq_len if seq_len > 0 else 0.0
                motif_positions[motif_name].append(norm_pos)

    n                = len(seqs)
    fractions        = {name: len(idx_set) / n
                        for name, idx_set in hits.items()}
    per_seq_n_motifs = [len(seq_motif_set[i]) for i in range(n)]
    per_seq_n_hits   = [seq_hit_count[i]       for i in range(n)]

    return fractions, per_seq_n_motifs, per_seq_n_hits, seq_motif_set, motif_positions

# ══════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════
print("="*50)
print("Loading sequences...")
gen_seqs = {}
for lbl, path in GEN_FILES.items():
    gen_seqs[lbl] = read_seqs(path)
    print(f"  {lbl}: n={len(gen_seqs[lbl])}")

print("\nLoading JASPAR motifs...")
motifs = load_motifs(JASPAR_MEME)

print("\nScanning motifs (this may take a few minutes)...")
frac_data      = {}
seq_n_motifs   = {}   # 每条序列命中的不同motif数
seq_n_hits     = {}   # 每条序列的总命中次数
seq_motif_sets = {}   # ← 新增
motif_pos = {}
for lbl in ["low", "mid", "high"]:
    print(f"  Scanning {lbl}...")
    (frac_data[lbl], seq_n_motifs[lbl], seq_n_hits[lbl],
     seq_motif_sets[lbl], motif_pos[lbl]) = \
        scan_sequences(gen_seqs[lbl], motifs)
    print(f"  Done. {len(frac_data[lbl])} motifs with at least one hit.")

# 保存额外统计
import pickle
with open(os.path.join(OUT_DIR, "seq_stats.pkl"), "wb") as f:
    pickle.dump({
        "seq_n_motifs":   seq_n_motifs,
        "seq_n_hits":     seq_n_hits,
        "seq_motif_sets": {          # ← 新增：每条序列命中的 motif 名称集合
            lbl: [seq_motif_sets[lbl][i] for i in range(len(gen_seqs[lbl]))]
            for lbl in ["low", "mid", "high"]
        },
        "motif_positions": motif_pos,  
    }, f)

# 构建频率表并保存（避免重复扫描）
all_tfs = sorted(set().union(*[set(d.keys()) for d in frac_data.values()]))
freq_df = pd.DataFrame(
    {lbl: {tf: frac_data[lbl].get(tf, 0.0) for tf in all_tfs}
     for lbl in ["low", "mid", "high"]}
)
freq_df.to_csv(os.path.join(OUT_DIR, "motif_freq_table.csv"))
print(f"\nSaved motif_freq_table.csv  shape={freq_df.shape}")
print("\nTop 20 by variance (条件间差异最大的motif):")
print(freq_df.assign(var=freq_df.std(axis=1))
             .sort_values("var", ascending=False)
             .head(20)[["low","mid","high","var"]]
             .round(3))

# ══════════════════════════════════════════════════════════════
# 5. 扫描真实序列（图3 生成 vs 真实 对比用）
# ══════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Scanning REAL sequences for Fig3...")

# 读取真实序列并按分位数分成 low/mid/high
real_df = pd.read_csv(REAL_CSV)
real_df.columns = [c.strip().lower() for c in real_df.columns]
real_df = real_df.dropna(subset=[SEQ_COL, ACTIVITY_COL])
real_df[ACTIVITY_COL] = pd.to_numeric(real_df[ACTIVITY_COL], errors="coerce")
real_df = real_df.dropna(subset=[ACTIVITY_COL])

# 用训练集相同的三等分分位数边界（与你训练时保持一致）
q33 = real_df[ACTIVITY_COL].quantile(1/3)
q67 = real_df[ACTIVITY_COL].quantile(2/3)
print(f"  Quantile boundaries: low<{q33:.4f}, mid<{q67:.4f}, high>={q67:.4f}")

def assign_label(val):
    if val < q33:  return "low"
    if val < q67:  return "mid"
    return "high"

real_df["label"] = real_df[ACTIVITY_COL].apply(assign_label)

# 每组最多取 1000 条（和生成序列数量量级对齐，加速扫描）
MAX_REAL = 1000
real_seqs_by_label = {}
for lbl in ["low", "mid", "high"]:
    subset = real_df[real_df["label"] == lbl][SEQ_COL].tolist()
    # 清洗序列
    cleaned = []
    for s in subset:
        s = "".join(c for c in str(s).upper() if c in "ACGT")
        if len(s) >= 20:
            cleaned.append(s)
    real_seqs_by_label[lbl] = cleaned[:MAX_REAL]
    print(f"  real {lbl}: n={len(real_seqs_by_label[lbl])}")

# FIMO 扫描真实序列
real_frac_data = {}
for lbl in ["low", "mid", "high"]:
    print(f"  Scanning real {lbl}...")
    fracs, _, _, _, _ = scan_sequences(real_seqs_by_label[lbl], motifs)
    real_frac_data[lbl] = fracs
    print(f"  Done. {len(fracs)} motifs hit.")

# 保存真实序列频率表
all_tfs_real = sorted(set().union(*[set(d.keys()) for d in real_frac_data.values()]))
real_freq_df = pd.DataFrame(
    {lbl: {tf: real_frac_data[lbl].get(tf, 0.0) for tf in all_tfs_real}
     for lbl in ["low", "mid", "high"]}
)
real_freq_df.to_csv(os.path.join(OUT_DIR, "motif_freq_table_real.csv"))
print(f"\nSaved motif_freq_table_real.csv  shape={real_freq_df.shape}")

# 保存真实序列每条序列的 motif 数量（FigI1 对比用）
real_seq_n_motifs = {}
for lbl in ["low", "mid", "high"]:
    print(f"  Scanning real {lbl} for per-seq stats...")
    _, per_seq_n, _, _, _ = scan_sequences(real_seqs_by_label[lbl], motifs)
    real_seq_n_motifs[lbl] = per_seq_n

with open(os.path.join(OUT_DIR, "real_seq_stats.pkl"), "wb") as f:
    pickle.dump({"seq_n_motifs": real_seq_n_motifs}, f)
print("Saved real_seq_stats.pkl")