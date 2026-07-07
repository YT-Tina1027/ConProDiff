import os, sys, gc, warnings, logging, pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import product, combinations
from scipy import stats
from scipy.linalg import sqrtm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

NUCLEOTIDES = ["A", "C", "G", "T"]
NUC_IDX     = {n: i for i, n in enumerate(NUCLEOTIDES)}

# ══════════════════════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════════════════════
GEN_LOW_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/low.txt"
GEN_MID_TXT  = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/mid.txt"
GEN_HIGH_TXT = "/home/yt/Code/DNA-Diffusion/data/outputs_yeast_CFG_gc2.0/high.txt"

REAL_CSV                = "/home/yt/Code/DNA-Diffusion/Data/SC_exp_short.csv"
ORACLE_DIR              = "/home/yt/Code/DNA-Diffusion/oracle"
ORACLE_MODEL_CONDITIONS = "defined_media"
JASPAR_MEME             = "/home/yt/Code/DNA-Diffusion/lunwen/JASPAR2026_Scerevisiae_CORE.meme"

OUT_DIR        = "/home/yt/Code/DNA-Diffusion/analysis_d3_eval1.0"
FIMO_CACHE_DIR = os.path.join(OUT_DIR, "fimo_cache")
ATTR_CACHE_DIR = os.path.join(OUT_DIR, "attr_cache")
for _d in [OUT_DIR, FIMO_CACHE_DIR, ATTR_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

LABEL_ORDER        = ["low", "mid", "high"]
SEQ_COL_CANDIDATES = ["sequence", "seq", "dna", "generated_sequence"]

N_ATTR_SEQS    = 2000
N_ATTR_REFS    = 5000
ATTR_KMER      = 6
MAX_REAL_FOR_PI = 5000


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def standardize_seq(s: str) -> str:
    s = str(s).upper().strip().replace("U", "T")
    return "".join(ch for ch in s if ch in "ATCG")

def find_seq_col(df: pd.DataFrame) -> str:
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到序列列，候选列: {SEQ_COL_CANDIDATES}")

def read_txt_sequences(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件: {path}")
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq = standardize_seq(line)
            if seq:
                seqs.append(seq)
    if not seqs:
        raise ValueError(f"文件中无有效序列: {path}")
    return seqs

def one_hot(seq: str, length: int) -> np.ndarray:
    oh = np.zeros((length, 4), dtype=np.float32)
    for i, c in enumerate(seq[:length]):
        if c in NUC_IDX:
            oh[i, NUC_IDX[c]] = 1.0
    return oh

def subsample(seqs: List[str], n: int) -> List[str]:
    if len(seqs) <= n:
        return seqs
    idx = np.random.choice(len(seqs), n, replace=False)
    return [seqs[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# RealOracle
# ══════════════════════════════════════════════════════════════════════════════
class RealOracle:
    ORACLE_LEN = 110

    def __init__(self, oracle_dir: str, model_conditions: str):
        self.oracle_dir       = oracle_dir
        self.model_conditions = model_conditions
        self._model  = None
        self._scaler = None
        self._batch  = None
        self._graph  = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import tensorflow as tf
        if self.oracle_dir not in sys.path:
            sys.path.insert(0, self.oracle_dir)
        from aux import load_model
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        gc.collect()
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._model, self._scaler, self._batch = load_model(self.model_conditions)

    def predict(self, seqs: List[str]) -> np.ndarray:
        self._ensure_loaded()
        if self.oracle_dir not in sys.path:
            sys.path.insert(0, self.oracle_dir)
        from aux import evaluate_model
        preds = evaluate_model(seqs, self._model, self._scaler, self._batch, self._graph)
        return np.asarray(preds, dtype=float).reshape(-1)

    def embed(self, seqs: List[str]) -> np.ndarray:
        k = 4
        all_kmers = ["".join(p) for p in product(NUCLEOTIDES, repeat=k)]
        km_idx    = {km: i for i, km in enumerate(all_kmers)}
        out = np.zeros((len(seqs), len(all_kmers)), dtype=np.float32)
        for r, seq in enumerate(seqs):
            seq = standardize_seq(seq)
            total = 0
            for i in range(len(seq) - k + 1):
                km = seq[i:i+k]
                if km in km_idx:
                    out[r, km_idx[km]] += 1
                    total += 1
            if total > 0:
                out[r] /= total
        return out

    def gradients(self, seqs: List[str], ref_seqs: List[str]) -> np.ndarray:
        self._ensure_loaded()
        import tensorflow as tf

        raw_len = len(standardize_seq(seqs[0]))

        def pad_right(s):
            s = standardize_seq(s)
            return (s + "A" * self.ORACLE_LEN)[:self.ORACLE_LEN]

        oh_seqs = np.stack([one_hot(pad_right(s), self.ORACLE_LEN) for s in seqs])
        oh_refs = np.stack([one_hot(pad_right(s), self.ORACLE_LEN) for s in ref_seqs])

        baseline  = oh_refs.mean(axis=0)
        diff      = oh_seqs - baseline[np.newaxis]
        all_attrs = np.zeros_like(oh_seqs)

        try:
            with self._graph.as_default():
                x_ph    = tf.placeholder(tf.float32, shape=[None, self.ORACLE_LEN, 4])
                output  = self._model(x_ph)
                grad_op = tf.gradients(tf.reduce_sum(output), x_ph)[0]
                sess    = tf.keras.backend.get_session()

                BATCH = 64
                grads_list = []
                for start in range(0, len(oh_seqs), BATCH):
                    g = sess.run(grad_op, feed_dict={x_ph: oh_seqs[start:start+BATCH]})
                    grads_list.append(g)
                grads = np.concatenate(grads_list, axis=0)

                all_attrs = grads * diff
                all_attrs -= all_attrs.mean(axis=-1, keepdims=True)

        except Exception as e:
            print(f"  [Attribution] TF1 梯度失败（{e}），返回零矩阵。")

        return all_attrs[:, :raw_len, :]


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def _parse_real_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "strength" not in df.columns:
        cands = ["activity", "score", "expression", "expr", "value"]
        found = next((c for c in cands if c in df.columns), None)
        if found is None:
            raise ValueError(f"缺少强度列，当前列: {list(df.columns)}")
        df["strength"] = df[found]
    seq_col = find_seq_col(df)
    df["sequence_std"] = df[seq_col].astype(str).apply(standardize_seq)
    df["strength"]     = pd.to_numeric(df["strength"], errors="coerce")
    df = df.dropna(subset=["strength", "sequence_std"])
    df = df[df["sequence_std"].str.len() > 0].reset_index(drop=True)
    if "label" not in df.columns:
        q1, q2 = df["strength"].quantile(1/3), df["strength"].quantile(2/3)
        df["label"] = df["strength"].apply(
            lambda x: "low" if x <= q1 else ("mid" if x <= q2 else "high"))
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df[df["label"].isin(LABEL_ORDER)].reset_index(drop=True)

def load_generated(low_txt, mid_txt, high_txt) -> Dict[str, List[str]]:
    return {b: read_txt_sequences(p)
            for b, p in [("low", low_txt), ("mid", mid_txt), ("high", high_txt)]}

def load_real(path: str) -> Dict[str, List[str]]:
    df = _parse_real_csv(path)
    return {b: df.loc[df["label"] == b, "sequence_std"].tolist() for b in LABEL_ORDER}

def load_real_strength(path: str) -> Dict[str, np.ndarray]:
    df = _parse_real_csv(path)
    return {b: df.loc[df["label"] == b, "strength"].values for b in LABEL_ORDER}


# ══════════════════════════════════════════════════════════════════════════════
# FimoScanner
# ══════════════════════════════════════════════════════════════════════════════
class FimoScanner:
    def __init__(self, meme_file: str, cache_dir: str, pvalue_thresh: float = 1e-3):
        self.meme_file     = meme_file
        self.cache_dir     = cache_dir
        self.pvalue_thresh = pvalue_thresh
        self._motifs       = None
        self._motif_names  = None

    def _ensure_motifs(self):
        if self._motifs is not None:
            return
        from pymemesuite.common import MotifFile
        all_m = []
        with MotifFile(self.meme_file) as f:
            for m in f:
                all_m.append(m)
        tf_groups = defaultdict(list)
        for m in all_m:
            tf_groups[m.name].append(m)
        def _ver(m):
            try: return int(m.accession.split(".")[-1])
            except: return 0
        self._motifs      = [max(ml, key=_ver) for ml in tf_groups.values()]
        self._motif_names = [
            (m.name.decode() if isinstance(m.name, bytes) else m.name)
            for m in self._motifs]
        print(f"  [FIMO] 去冗余后 motif 数: {len(self._motifs)}")

    def _scan(self, seqs: List[str]) -> np.ndarray:
        from pymemesuite.common import Sequence, Background, Alphabet, Array
        from pymemesuite.fimo import FIMO
        self._ensure_motifs()
        matrix   = np.zeros((len(seqs), len(self._motifs)), dtype=np.float32)
        seq_objs = [Sequence(s, name=f"seq_{i}".encode()) for i, s in enumerate(seqs)]
        alphabet  = Alphabet.dna()
        background = Background(alphabet, Array([0.25, 0.25, 0.25, 0.25]))
        fimo = FIMO(both_strands=True, threshold=self.pvalue_thresh,
                    max_stored_scores=200000)
        for mi, motif in enumerate(self._motifs):
            pattern = fimo.score_motif(motif, seq_objs, background)
            for el in pattern.matched_elements:
                if el.pvalue is not None and el.pvalue <= self.pvalue_thresh:
                    src = el.source.name
                    if isinstance(src, bytes): src = src.decode()
                    matrix[int(src.replace("seq_", "")), mi] += 1
        return matrix

    def hit_matrix(self, seqs: List[str], tag: str) -> np.ndarray:
        cache = os.path.join(self.cache_dir, f"{tag}_n{len(seqs)}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                data = pickle.load(f)
            self._ensure_motifs()
            print(f"  [FIMO] 缓存命中: {tag}  shape={data['matrix'].shape}")
            return data["matrix"]
        print(f"  [FIMO] 扫描: {tag}  n={len(seqs)} ...")
        mat = self._scan(seqs)
        with open(cache, "wb") as f:
            pickle.dump({"matrix": mat, "motif_names": self._motif_names}, f)
        print(f"  [FIMO] 完成: {tag}  hits={mat.sum():.0f}")
        return mat


# ══════════════════════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════════════════════
def conditional_generation_fidelity(gen_seqs, real_strength, oracle):
    pred_gen = oracle.predict(gen_seqs)
    return float((pred_gen.mean() - real_strength.mean()) ** 2)

def frechet_distance(real_seqs, gen_seqs, oracle):
    emb_r = oracle.embed(real_seqs)
    emb_g = oracle.embed(gen_seqs)
    mu1, mu2 = emb_r.mean(0), emb_g.mean(0)
    diff = mu1 - mu2
    eps  = np.eye(emb_r.shape[1]) * 1e-4
    s1   = np.cov(emb_r.T) + eps
    s2   = np.cov(emb_g.T) + eps
    cm   = sqrtm(s1 @ s2)
    if np.iscomplexobj(cm): cm = cm.real
    raw  = float(diff @ diff + np.trace(s1 + s2 - 2 * cm))
    return max(0.0, raw)

def predictive_distribution_shift(real_seqs, gen_seqs, oracle):
    ks, _ = stats.ks_2samp(oracle.predict(real_seqs), oracle.predict(gen_seqs))
    return float(ks)

def percent_identity(query_seqs, ref_seqs) -> Tuple[float, float]:
    ref_arr = np.array([[c for c in s] for s in ref_seqs])
    max_ids = []
    for qa in query_seqs:
        q_arr = np.array([c for c in qa])
        pis   = (ref_arr == q_arr).mean(axis=1)
        max_ids.append(pis.max())
    return float(np.mean(max_ids)), float(max(max_ids))

def kmer_spectrum_shift(real_seqs, gen_seqs, k_range=range(1, 8)):
    def _freq(seqs, k, kmers):
        cnt = Counter()
        for s in seqs:
            for i in range(len(s) - k + 1): cnt[s[i:i+k]] += 1
        total = sum(cnt.values()) + 1e-10
        return np.array([cnt[km] / total for km in kmers])

    def _js(p, q):
        m = (p + q) / 2
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
        return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))

    return float(np.mean([
        _js(_freq(real_seqs, k, ["".join(t) for t in product(NUCLEOTIDES, repeat=k)]),
            _freq(gen_seqs,  k, ["".join(t) for t in product(NUCLEOTIDES, repeat=k)]))
        for k in k_range]))

def discriminatability(real_seqs, gen_seqs, oracle, seed=0):
    X = np.vstack([oracle.embed(real_seqs), oracle.embed(gen_seqs)])
    y = np.array([1] * len(real_seqs) + [0] * len(gen_seqs))
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

def motif_enrichment(real_seqs, gen_seqs, scanner, real_tag, gen_tag):
    hits_r = scanner.hit_matrix(real_seqs, real_tag).sum(0) / len(real_seqs)
    hits_g = scanner.hit_matrix(gen_seqs,  gen_tag).sum(0) / len(gen_seqs)
    mask   = (hits_r > 0) | (hits_g > 0)
    if mask.sum() < 2: return float("nan")
    r, _ = stats.pearsonr(hits_r[mask], hits_g[mask])
    return float(r)

def motif_cooccurrence(real_seqs, gen_seqs, scanner, real_tag, gen_tag):
    mat_r  = scanner.hit_matrix(real_seqs, real_tag)
    mat_g  = scanner.hit_matrix(gen_seqs,  gen_tag)
    active = (mat_r.sum(0) > 0) | (mat_g.sum(0) > 0)
    mat_r, mat_g = mat_r[:, active], mat_g[:, active]
    if mat_r.shape[1] < 2: return float("nan")
    eps = np.eye(mat_r.shape[1]) * 1e-6
    return float(np.linalg.norm(
        (np.cov(mat_r.T) + eps) - (np.cov(mat_g.T) + eps), "fro"))

def _attr_kld(attr_maps: np.ndarray, k: int = ATTR_KMER) -> float:
    if np.all(attr_maps == 0):
        return float("nan")
    pos_scores = np.abs(attr_maps).sum(axis=-1)
    kmer_cnt, kmer_total = Counter(), 0
    for i in range(len(attr_maps)):
        score    = pos_scores[i]
        thresh   = np.percentile(score, 75)
        high_pos = np.where(score >= thresh)[0]
        L        = attr_maps.shape[1]
        for p in high_pos:
            start, end = max(0, p - k // 2), min(L, p - k // 2 + k)
            if end - start == k:
                indices = [int(np.argmax(attr_maps[i, pos])) for pos in range(start, end)]
                kmer    = "".join(NUCLEOTIDES[idx] for idx in indices)
                if all(c in "ATCG" for c in kmer):
                    kmer_cnt[kmer] += 1
                    kmer_total     += 1
    if kmer_total == 0:
        return float("nan")
    uniform = 1.0 / (4 ** k)
    kld = sum(cnt / kmer_total * np.log(cnt / kmer_total / uniform)
              for cnt in kmer_cnt.values())
    return float(kld)

def attribution_consistency(seqs, oracle, n_top=N_ATTR_SEQS,
                            n_refs=N_ATTR_REFS, cache_tag="") -> float:
    cache_path = os.path.join(ATTR_CACHE_DIR, f"{cache_tag}.pkl") if cache_tag else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            attr_maps = pickle.load(f)
        print(f"  [Attr] 缓存命中: {cache_tag}  shape={attr_maps.shape}")
        return _attr_kld(attr_maps)

    print(f"  [Attr] 预测活性排序 n={len(seqs)} ...")
    preds    = oracle.predict(seqs)
    top_seqs = [seqs[i] for i in np.argsort(preds)[-n_top:]]
    refs     = ["".join(np.random.choice(NUCLEOTIDES, RealOracle.ORACLE_LEN))
                for _ in range(n_refs)]

    print(f"  [Attr] GradientSHAP  n_top={len(top_seqs)}, n_refs={n_refs} ...")
    attr_maps = oracle.gradients(top_seqs, refs)

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(attr_maps, f)

    return _attr_kld(attr_maps)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)

    print("加载数据 ...")
    gen_seqs_dict      = load_generated(GEN_LOW_TXT, GEN_MID_TXT, GEN_HIGH_TXT)
    real_seqs_dict     = load_real(REAL_CSV)
    real_strength_dict = load_real_strength(REAL_CSV)

    for b in LABEL_ORDER:
        print(f"  {b}: real={len(real_seqs_dict[b])}  gen={len(gen_seqs_dict[b])}"
              f"  strength_mean={real_strength_dict[b].mean():.3f}")

    oracle  = RealOracle(ORACLE_DIR, ORACLE_MODEL_CONDITIONS)
    scanner = FimoScanner(JASPAR_MEME, FIMO_CACHE_DIR, pvalue_thresh=1e-3)

    # 合并所有 bin
    all_real = sum([real_seqs_dict[b] for b in LABEL_ORDER], [])
    all_gen  = sum([gen_seqs_dict[b]  for b in LABEL_ORDER], [])
    all_str  = np.concatenate([real_strength_dict[b] for b in LABEL_ORDER])
    real_sub = subsample(all_real, MAX_REAL_FOR_PI)

    print(f"\n合并后: real={len(all_real)}  gen={len(all_gen)}")

    m = {}
    m["MSE ↓"]        = round(conditional_generation_fidelity(all_gen, all_str, oracle), 6)
    m["KS stat ↓"]    = round(predictive_distribution_shift(all_real, all_gen, oracle), 3)
    m["Fréchet ↓"]    = round(frechet_distance(real_sub, all_gen, oracle), 4)

    pi_avg, pi_max    = percent_identity(all_gen, real_sub)
    m["PI avg %"]     = round(pi_avg * 100, 1)
    m["PI max %"]     = round(pi_max * 100, 1)
    m["kmer JS ↓"]    = round(kmer_spectrum_shift(all_real, all_gen), 4)
    m["AUROC→0.5"]    = round(discriminatability(all_real, all_gen, oracle), 3)

    m["Motif r ↑"]    = round(motif_enrichment(
                            all_real, all_gen, scanner, "real_D3", "gen_D3"), 3)
    m["Motif Frob ↓"] = round(motif_cooccurrence(
                            all_real, all_gen, scanner, "real_D3", "gen_D3"), 2)
    m["Attr KLD ↑"]   = round(attribution_consistency(
                            all_gen, oracle, cache_tag="attr_D3_gen"), 3)

    df = pd.DataFrame([m], index=["D3"])
    print("\n" + "═" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df.to_string())

    out_csv = os.path.join(OUT_DIR, "d3_eval_results.csv")
    df.to_csv(out_csv)
    print(f"\n结果已保存: {out_csv}")


if __name__ == "__main__":
    main()