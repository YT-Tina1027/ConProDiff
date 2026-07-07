import os
import re
import numpy as np
import pandas as pd

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# =========================
# 路径配置
# =========================
FIMO_BASE = "/home/yt/Code/DNA-Diffusion/motif_fimo_results"
FASTA_SUMMARY_CSV = "/home/yt/Code/DNA-Diffusion/Data/motif_analysis_fastas/fasta_export_summary.csv"
SUMMARY_CSV = "/home/yt/Code/DATA/yetfasco_expert_curated_summary.csv"

OUT_DIR = "/home/yt/Code/DNA-Diffusion/motif_bin_specific_selection"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# FIMO 文件
# =========================
GROUP_TO_FIMO = {
    "real_low":  os.path.join(FIMO_BASE, "fimo_real_low",  "fimo.tsv"),
    "real_mid":  os.path.join(FIMO_BASE, "fimo_real_mid",  "fimo.tsv"),
    "real_high": os.path.join(FIMO_BASE, "fimo_real_high", "fimo.tsv"),
    "gen_low":   os.path.join(FIMO_BASE, "fimo_gen_low",   "fimo.tsv"),
    "gen_mid":   os.path.join(FIMO_BASE, "fimo_gen_mid",   "fimo.tsv"),
    "gen_high":  os.path.join(FIMO_BASE, "fimo_gen_high",  "fimo.tsv"),
}

LABELS = ["low", "mid", "high"]

# =========================
# 参数
# =========================
USE_QVALUE_FILTER = True
QVALUE_THRESHOLD = 0.05

MIN_TARGET_FRACTION = 0.01    # 真实目标 bin 中至少 1% 序列命中
MIN_OTHER_FRACTION = 0.0
FDR_THRESHOLD = 0.05
MIN_LOG2FC = 0.0              # 只保留正富集
TOP_N_PER_BIN = 8

EPS = 1e-9

# =========================
# 工具函数
# =========================
def simplify_motif_name(name: str) -> str:
    name = str(name)
    name = re.sub(r'_\d+$', '', name)
    return name

def load_motif_name_map(summary_csv):
    df = pd.read_csv(summary_csv)
    if "motif_name" not in df.columns:
        raise ValueError("summary csv 缺少 motif_name 列")
    df = df.copy()
    df["display_name"] = df["motif_name"].apply(simplify_motif_name)
    return dict(zip(df["motif_name"], df["display_name"]))

def load_total_counts(fasta_summary_csv):
    df = pd.read_csv(fasta_summary_csv)
    out = {}
    for _, row in df.iterrows():
        out[row["group"]] = int(row["n_sequences"])
    return out

def read_fimo_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 FIMO 结果: {path}")

    df = pd.read_csv(path, sep="\t", comment="#")
    if len(df) == 0:
        return df

    for c in ["motif_id", "sequence_name"]:
        if c not in df.columns:
            raise ValueError(f"{path} 缺少列: {c}")

    if USE_QVALUE_FILTER and ("q-value" in df.columns):
        df = df[df["q-value"] <= QVALUE_THRESHOLD].copy()

    return df.reset_index(drop=True)

def get_unique_hit_sequences_per_motif(fimo_df):
    """
    返回:
      motif_id -> set(sequence_name)
    """
    if len(fimo_df) == 0:
        return {}

    grouped = (
        fimo_df[["motif_id", "sequence_name"]]
        .drop_duplicates()
        .groupby("motif_id")["sequence_name"]
        .apply(set)
    )
    return grouped.to_dict()

def union_sets(dicts, motif_id):
    out = set()
    for d in dicts:
        if motif_id in d:
            out |= d[motif_id]
    return out

def safe_log2fc(p1, p2):
    return np.log2((p1 + EPS) / (p2 + EPS))

# =========================
# 构建每组 motif -> hit seq set
# =========================
def build_group_hit_maps():
    group_hit_maps = {}
    for group, path in GROUP_TO_FIMO.items():
        fimo_df = read_fimo_tsv(path)
        group_hit_maps[group] = get_unique_hit_sequences_per_motif(fimo_df)
    return group_hit_maps

# =========================
# 每个 bin 做 real-only enrichment
# =========================
def test_bin_specific_motifs(bin_name, group_hit_maps, total_counts, motif_name_map):
    """
    以真实数据做:
      target = real_bin
      background = other two real bins
    """
    target_group = f"real_{bin_name}"
    other_groups = [f"real_{x}" for x in LABELS if x != bin_name]

    n_target = total_counts[target_group]
    n_other = sum(total_counts[g] for g in other_groups)

    target_map = group_hit_maps[target_group]
    other_maps = [group_hit_maps[g] for g in other_groups]

    all_motifs = set(target_map.keys())
    for om in other_maps:
        all_motifs |= set(om.keys())

    rows = []

    for motif_id in all_motifs:
        target_hits = len(target_map.get(motif_id, set()))
        other_hits = len(union_sets(other_maps, motif_id))

        a = target_hits
        b = n_target - target_hits
        c = other_hits
        d = n_other - other_hits

        # 保险
        a = max(a, 0)
        b = max(b, 0)
        c = max(c, 0)
        d = max(d, 0)

        # Fisher: target 是否富集
        odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

        frac_target = a / max(n_target, 1)
        frac_other = c / max(n_other, 1)
        log2fc = safe_log2fc(frac_target, frac_other)

        rows.append({
            "bin": bin_name,
            "motif_id": motif_id,
            "display_name": motif_name_map.get(motif_id, simplify_motif_name(motif_id)),
            "target_group": target_group,
            "other_groups": "+".join(other_groups),
            "target_hits": a,
            "target_total": n_target,
            "target_fraction": frac_target,
            "other_hits": c,
            "other_total": n_other,
            "other_fraction": frac_other,
            "odds_ratio": odds_ratio,
            "p_value": pval,
            "log2_fc": log2fc,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # FDR
    reject, qvals, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["q_value"] = qvals
    df["fdr_pass"] = reject

    # 过滤
    df_filt = df[
        (df["q_value"] < FDR_THRESHOLD) &
        (df["log2_fc"] > MIN_LOG2FC) &
        (df["target_fraction"] >= MIN_TARGET_FRACTION) &
        (df["other_fraction"] >= MIN_OTHER_FRACTION)
    ].copy()

    # 排序：先显著性，再效应量，再 target fraction
    df_filt = df_filt.sort_values(
        by=["q_value", "log2_fc", "target_fraction"],
        ascending=[True, False, False]
    )

    # 取 top N
    df_top = df_filt.head(TOP_N_PER_BIN).copy()

    return df, df_filt, df_top

# =========================
# 评估这些 real-defined motifs 在 gen 中是否复现
# =========================
def evaluate_selected_motifs_in_generated(bin_name, top_df, group_hit_maps, total_counts):
    gen_group = f"gen_{bin_name}"
    other_gen_groups = [f"gen_{x}" for x in LABELS if x != bin_name]

    n_gen = total_counts[gen_group]
    n_other_gen = sum(total_counts[g] for g in other_gen_groups)

    gen_map = group_hit_maps[gen_group]
    other_gen_maps = [group_hit_maps[g] for g in other_gen_groups]

    rows = []
    for _, row in top_df.iterrows():
        motif_id = row["motif_id"]

        gen_hits = len(gen_map.get(motif_id, set()))
        other_gen_hits = len(union_sets(other_gen_maps, motif_id))

        frac_gen = gen_hits / max(n_gen, 1)
        frac_other_gen = other_gen_hits / max(n_other_gen, 1)

        rows.append({
            "bin": bin_name,
            "motif_id": motif_id,
            "display_name": row["display_name"],
            "real_target_fraction": row["target_fraction"],
            "real_other_fraction": row["other_fraction"],
            "real_log2_fc": row["log2_fc"],
            "real_q_value": row["q_value"],
            "gen_target_fraction": frac_gen,
            "gen_other_fraction": frac_other_gen,
            "gen_log2_fc": safe_log2fc(frac_gen, frac_other_gen),
        })

    return pd.DataFrame(rows)

# =========================
# 主函数
# =========================
def main():
    motif_name_map = load_motif_name_map(SUMMARY_CSV)
    total_counts = load_total_counts(FASTA_SUMMARY_CSV)
    group_hit_maps = build_group_hit_maps()

    all_full = []
    all_filt = []
    all_top = []
    all_eval = []

    for bin_name in LABELS:
        df_full, df_filt, df_top = test_bin_specific_motifs(
            bin_name=bin_name,
            group_hit_maps=group_hit_maps,
            total_counts=total_counts,
            motif_name_map=motif_name_map
        )

        eval_df = evaluate_selected_motifs_in_generated(
            bin_name=bin_name,
            top_df=df_top,
            group_hit_maps=group_hit_maps,
            total_counts=total_counts
        )

        all_full.append(df_full)
        all_filt.append(df_filt)
        all_top.append(df_top)
        all_eval.append(eval_df)

        df_full.to_csv(os.path.join(OUT_DIR, f"{bin_name}_all_motifs_stats.csv"), index=False)
        df_filt.to_csv(os.path.join(OUT_DIR, f"{bin_name}_filtered_motifs.csv"), index=False)
        df_top.to_csv(os.path.join(OUT_DIR, f"{bin_name}_top_motifs.csv"), index=False)
        eval_df.to_csv(os.path.join(OUT_DIR, f"{bin_name}_top_motifs_eval_in_generated.csv"), index=False)

        print(f"[{bin_name}] all={len(df_full)} filtered={len(df_filt)} top={len(df_top)}")

    pd.concat(all_full, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_all_motifs_stats.csv"), index=False
    )
    pd.concat(all_filt, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_filtered_motifs.csv"), index=False
    )
    pd.concat(all_top, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_top_motifs.csv"), index=False
    )
    pd.concat(all_eval, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "all_bins_top_motifs_eval_in_generated.csv"), index=False
    )

    print(f"\nSaved results to: {OUT_DIR}")

if __name__ == "__main__":
    main()