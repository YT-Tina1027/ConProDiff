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
# 路径
# ══════════════════════════════════════════════════════════════
JASPAR_MEME = "/home/yt/Code/DNA-Diffusion/lunwen/JASPAR2026_Scerevisiae_CORE.meme"

GEN_FILES = {
    "low":  "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/low.txt",
    "mid":  "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/mid.txt",
    "high": "/home/yt/Code/DNA-Diffusion/result/experiments_SC/data_cfg2.0/high.txt",
}

OUT_DIR  = "/home/yt/Code/DNA-Diffusion/lunwen/tu3_SC"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 真实序列路径 ──────────────────────────────────────────────
REAL_CSV     = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
SEQ_COL      = "sequence"
ACTIVITY_COL = "strength"

# ══════════════════════════════════════════════════════════════
# 1. 读取序列
# ══════════════════════════════════════════════════════════════
def read_seqs(path):
    seqs = []
    for line in open(path):
        tok = max(re.split(r"[\s,\t;|]+", line.strip()),
                  key=lambda x: len(x), default="")
        s = "".join(c for c in tok.upper() if c in "ACGT")
        if len(s) >= 20:
            seqs.append(s)
    return seqs

# ══════════════════════════════════════════════════════════════
# 2. 加载 MEME 文件（用 accession 去冗余，与大肠杆菌版一致）
# ══════════════════════════════════════════════════════════════
def load_motifs(meme_file):
    all_motifs = []
    with MotifFile(meme_file) as f:
        for m in f:
            all_motifs.append(m)
    print(f"  原始motif数量: {len(all_motifs)}")

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
# 3. TF名称解析（JASPAR accession 格式如 MA0265.3）
# ══════════════════════════════════════════════════════════════
def parse_tf_name(m):
    """优先用 motif.name，fallback 到 accession"""
    name = m.name
    if isinstance(name, bytes):
        name = name.decode()
    return name if name else (m.accession.decode() if m.accession else "unknown")

# ══════════════════════════════════════════════════════════════
# 4. FIMO 扫描（与大肠杆菌版结构完全一致）
# ══════════════════════════════════════════════════════════════
def scan_sequences(seqs, motifs, pvalue_thresh=1e-3):
    seq_objs   = [Sequence(s, name=f"seq_{i}".encode())
                  for i, s in enumerate(seqs)]
    alphabet   = Alphabet.dna()
    background = Background(alphabet, Array([0.25, 0.25, 0.25, 0.25]))
    fimo       = FIMO(both_strands=True, threshold=pvalue_thresh,
                      max_stored_scores=100000)

    hits            = defaultdict(set)
    seq_motif_set   = defaultdict(set)
    seq_hit_count   = defaultdict(int)
    motif_positions = defaultdict(list)

    for m in motifs:
        motif_name = parse_tf_name(m)

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
                seq_len  = len(seqs[idx])
                norm_pos = el.start / seq_len if seq_len > 0 else 0.0
                motif_positions[motif_name].append(norm_pos)

    n                = len(seqs)
    fractions        = {name: len(idx_set) / n
                        for name, idx_set in hits.items()}
    per_seq_n_motifs = [len(seq_motif_set[i]) for i in range(n)]
    per_seq_n_hits   = [seq_hit_count[i]       for i in range(n)]

    return fractions, per_seq_n_motifs, per_seq_n_hits, seq_motif_set, motif_positions

# ══════════════════════════════════════════════════════════════
# 5. 主流程 — 扫描生成序列
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
seq_n_motifs   = {}
seq_n_hits     = {}
seq_motif_sets = {}
motif_pos      = {}

for lbl in ["low", "mid", "high"]:
    print(f"  Scanning {lbl}...")
    (frac_data[lbl], seq_n_motifs[lbl], seq_n_hits[lbl],
     seq_motif_sets[lbl], motif_pos[lbl]) = \
        scan_sequences(gen_seqs[lbl], motifs)
    print(f"  Done. {len(frac_data[lbl])} motifs with at least one hit.")

with open(os.path.join(OUT_DIR, "seq_stats.pkl"), "wb") as f:
    pickle.dump({
        "seq_n_motifs":   seq_n_motifs,
        "seq_n_hits":     seq_n_hits,
        "seq_motif_sets": {
            lbl: [seq_motif_sets[lbl][i] for i in range(len(gen_seqs[lbl]))]
            for lbl in ["low", "mid", "high"]
        },
        "motif_positions": motif_pos,
    }, f)

all_tfs = sorted(set().union(*[set(d.keys()) for d in frac_data.values()]))
freq_df = pd.DataFrame(
    {lbl: {tf: frac_data[lbl].get(tf, 0.0) for tf in all_tfs}
     for lbl in ["low", "mid", "high"]}
)
freq_df.to_csv(os.path.join(OUT_DIR, "motif_freq_table.csv"))
print(f"\nSaved motif_freq_table.csv  shape={freq_df.shape}")
print("\nTop 20 by variance:")
print(freq_df.assign(var=freq_df.std(axis=1))
             .sort_values("var", ascending=False)
             .head(20)[["low","mid","high","var"]]
             .round(3))

# ══════════════════════════════════════════════════════════════
# 6. 扫描真实序列（图3 生成 vs 真实 对比用）
# ══════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Scanning REAL sequences for Fig3...")

real_df = pd.read_csv(REAL_CSV)
real_df.columns = [c.strip().lower() for c in real_df.columns]
real_df = real_df.dropna(subset=[SEQ_COL, ACTIVITY_COL])
real_df[ACTIVITY_COL] = pd.to_numeric(real_df[ACTIVITY_COL], errors="coerce")
real_df = real_df.dropna(subset=[ACTIVITY_COL])

q33 = real_df[ACTIVITY_COL].quantile(1/3)
q67 = real_df[ACTIVITY_COL].quantile(2/3)
print(f"  Quantile boundaries: low<{q33:.4f}, mid<{q67:.4f}, high>={q67:.4f}")

def assign_label(val):
    if val < q33: return "low"
    if val < q67: return "mid"
    return "high"

real_df["label"] = real_df[ACTIVITY_COL].apply(assign_label)

MAX_REAL = 1000
real_seqs_by_label = {}
for lbl in ["low", "mid", "high"]:
    subset = real_df[real_df["label"] == lbl][SEQ_COL].tolist()
    cleaned = []
    for s in subset:
        s = "".join(c for c in str(s).upper() if c in "ACGT")
        if len(s) >= 20:
            cleaned.append(s)
    real_seqs_by_label[lbl] = cleaned[:MAX_REAL]
    print(f"  real {lbl}: n={len(real_seqs_by_label[lbl])}")

real_frac_data = {}
for lbl in ["low", "mid", "high"]:
    print(f"  Scanning real {lbl}...")
    fracs, _, _, _, _ = scan_sequences(real_seqs_by_label[lbl], motifs)
    real_frac_data[lbl] = fracs
    print(f"  Done. {len(fracs)} motifs hit.")

all_tfs_real = sorted(set().union(*[set(d.keys()) for d in real_frac_data.values()]))
real_freq_df = pd.DataFrame(
    {lbl: {tf: real_frac_data[lbl].get(tf, 0.0) for tf in all_tfs_real}
     for lbl in ["low", "mid", "high"]}
)
real_freq_df.to_csv(os.path.join(OUT_DIR, "motif_freq_table_real.csv"))
print(f"\nSaved motif_freq_table_real.csv  shape={real_freq_df.shape}")

real_seq_n_motifs = {}
for lbl in ["low", "mid", "high"]:
    print(f"  Scanning real {lbl} for per-seq stats...")
    _, per_seq_n, _, _, _ = scan_sequences(real_seqs_by_label[lbl], motifs)
    real_seq_n_motifs[lbl] = per_seq_n

with open(os.path.join(OUT_DIR, "real_seq_stats.pkl"), "wb") as f:
    pickle.dump({"seq_n_motifs": real_seq_n_motifs}, f)
print("Saved real_seq_stats.pkl")