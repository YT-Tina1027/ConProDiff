import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats import fisher_exact

# =========================
# 路径
# =========================
FASTA_DIR = "/home/yt/Code/DNA-Diffusion/Data/motif_analysis_fastas"
OUT_DIR = "/home/yt/Code/DNA-Diffusion/kmer6_bin_specific_selection"
os.makedirs(OUT_DIR, exist_ok=True)

REAL_FASTAS = {
    "low":  os.path.join(FASTA_DIR, "real_low.fa"),
    "mid":  os.path.join(FASTA_DIR, "real_mid.fa"),
    "high": os.path.join(FASTA_DIR, "real_high.fa"),
}

GEN_FASTAS = {
    "low":  os.path.join(FASTA_DIR, "gen_low.fa"),
    "mid":  os.path.join(FASTA_DIR, "gen_mid.fa"),
    "high": os.path.join(FASTA_DIR, "gen_high.fa"),
}

LABELS = ["low", "mid", "high"]

# =========================
# 参数
# =========================
K = 6
MIN_TARGET_FRACTION = 0.01
FDR_THRESHOLD = 0.05
MIN_LOG2FC = 0.0
TOP_N_PER_BIN = 12
EPS = 1e-12

# =========================
# 基础函数
# =========================
def read_fasta(path):
    seqs = []
    seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    seqs.append("".join(seq).upper())
                    seq = []
            else:
                seq.append(line.strip())
        if seq:
            seqs.append("".join(seq).upper())
    return seqs

def standardize_seq(s):
    s = str(s).upper().strip().replace("U", "T")
    return "".join([x for x in s if x in "ACGT"])

def all_kmers(k):
    return ["".join(x) for x in itertools.product("ACGT", repeat=k)]

def get_kmers_in_seq(seq, k):
    seq = standardize_seq(seq)
    out = set()
    if len(seq) < k:
        return out
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if set(kmer).issubset({"A", "C", "G", "T"}):
            out.add(kmer)
    return out

def build_presence_matrix(seqs, k):
    """
    返回:
      presence_counts: dict[kmer] = number of sequences containing this kmer
      total_n: total number of sequences
    """
    kmers = all_kmers(k)
    counts = dict.fromkeys(kmers, 0)

    for seq in seqs:
        present = get_kmers_in_seq(seq, k)
        for km in present:
            counts[km] += 1

    return counts, len(seqs)

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    qvals = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * n / rank
        q = min(q, prev)
        prev = q
        qvals[i] = q

    out = np.empty(n, dtype=float)
    out[order] = qvals
    reject = out < FDR_THRESHOLD
    return reject, out

def safe_log2fc(a, b):
    return np.log2((a + EPS) / (b + EPS))

# =========================
# 真实数据定义 bin-specific 6-mers
# =========================
def test_bin_specific_kmers(bin_name, real_presence, real_totals):
    target_counts = real_presence[bin_name]
    target_total = real_totals[bin_name]

    other_bins = [x for x in LABELS if x != bin_name]
    other_total = sum(real_totals[x] for x in other_bins)

    rows = []
    for km in all_kmers(K):
        a = target_counts.get(km, 0)
        b = target_total - a
        c = sum(real_presence[x].get(km, 0) for x in other_bins)
        d = other_total - c

        odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

        frac_target = a / max(target_total, 1)
        frac_other = c / max(other_total, 1)
        log2_fc = safe_log2fc(frac_target, frac_other)

        rows.append({
            "bin": bin_name,
            "kmer": km,
            "target_hits": a,
            "target_total": target_total,
            "target_fraction": frac_target,
            "other_hits": c,
            "other_total": other_total,
            "other_fraction": frac_other,
            "odds_ratio": odds_ratio,
            "p_value": pval,
            "log2_fc": log2_fc
        })

    df = pd.DataFrame(rows)
    reject, qvals = bh_fdr(df["p_value"].values)
    df["q_value"] = qvals
    df["fdr_pass"] = reject

    df_filt = df[
        (df["q_value"] < FDR_THRESHOLD) &
        (df["target_fraction"] >= MIN_TARGET_FRACTION) &
        (df["log2_fc"] > MIN_LOG2FC)
    ].copy()

    df_filt = df_filt.sort_values(
        by=["q_value", "log2_fc", "target_fraction"],
        ascending=[True, False, False]
    )

    df_top = df_filt.head(TOP_N_PER_BIN).copy()
    return df, df_filt, df_top

# =========================
# 在生成数据里验证
# =========================
def evaluate_top_kmers_in_generated(bin_name, top_df, gen_presence, gen_totals):
    target_counts = gen_presence[bin_name]
    target_total = gen_totals[bin_name]

    other_bins = [x for x in LABELS if x != bin_name]
    other_total = sum(gen_totals[x] for x in other_bins)

    rows = []
    for _, row in top_df.iterrows():
        km = row["kmer"]

        a = target_counts.get(km, 0)
        c = sum(gen_presence[x].get(km, 0) for x in other_bins)

        frac_target = a / max(target_total, 1)
        frac_other = c / max(other_total, 1)

        rows.append({
            "bin": bin_name,
            "kmer": km,
            "real_target_fraction": row["target_fraction"],
            "real_other_fraction": row["other_fraction"],
            "real_log2_fc": row["log2_fc"],
            "real_q_value": row["q_value"],
            "gen_target_fraction": frac_target,
            "gen_other_fraction": frac_other,
            "gen_log2_fc": safe_log2fc(frac_target, frac_other)
        })

    return pd.DataFrame(rows)

# =========================
# 主函数
# =========================
def main():
    real_seqs = {lab: [standardize_seq(s) for s in read_fasta(REAL_FASTAS[lab])] for lab in LABELS}
    gen_seqs  = {lab: [standardize_seq(s) for s in read_fasta(GEN_FASTAS[lab])]  for lab in LABELS}

    real_presence = {}
    real_totals = {}
    gen_presence = {}
    gen_totals = {}

    print("Building real 6-mer presence...")
    for lab in LABELS:
        counts, total = build_presence_matrix(real_seqs[lab], K)
        real_presence[lab] = counts
        real_totals[lab] = total
        print(f"real_{lab}: {total}")

    print("Building generated 6-mer presence...")
    for lab in LABELS:
        counts, total = build_presence_matrix(gen_seqs[lab], K)
        gen_presence[lab] = counts
        gen_totals[lab] = total
        print(f"gen_{lab}: {total}")

    all_full = []
    all_filt = []
    all_top = []
    all_eval = []

    for bin_name in LABELS:
        df_full, df_filt, df_top = test_bin_specific_kmers(
            bin_name, real_presence, real_totals
        )
        df_eval = evaluate_top_kmers_in_generated(
            bin_name, df_top, gen_presence, gen_totals
        )

        df_full.to_csv(os.path.join(OUT_DIR, f"{bin_name}_all_6mer_stats.csv"), index=False)
        df_filt.to_csv(os.path.join(OUT_DIR, f"{bin_name}_filtered_6mers.csv"), index=False)
        df_top.to_csv(os.path.join(OUT_DIR, f"{bin_name}_top_6mers.csv"), index=False)
        df_eval.to_csv(os.path.join(OUT_DIR, f"{bin_name}_top_6mers_eval_in_generated.csv"), index=False)

        all_full.append(df_full)
        all_filt.append(df_filt)
        all_top.append(df_top)
        all_eval.append(df_eval)

        print(f"[{bin_name}] filtered={len(df_filt)} top={len(df_top)}")

    pd.concat(all_full, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_all_6mer_stats.csv"), index=False
    )
    pd.concat(all_filt, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_filtered_6mers.csv"), index=False
    )
    pd.concat(all_top, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_top_6mers.csv"), index=False
    )
    pd.concat(all_eval, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_top_6mers_eval_in_generated.csv"), index=False
    )

    print(f"\nSaved all results to: {OUT_DIR}")

if __name__ == "__main__":
    main()