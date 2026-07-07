import os
import re
import pickle
from itertools import product
from typing import Dict, List, Optional

import editdistance
import numpy as np
import pandas as pd
import torch
import tempfile, subprocess
from dnadiffusion.utils.utils import convert_to_seq
from Predictor.predictor_models import LSTMModel


# ---------------------------------------------------------------------------
# Threshold loader (shared)
# ---------------------------------------------------------------------------

def load_eval_thresholds_from_bin_edges(config: Dict):
    """
    从训练数据处理保存的 pkl 中读取 bin_edges，保证 predictor 评估分箱与训练分箱一致。
    适用于 num_bins=3: bin_edges=[-inf, q1, q2, inf]
    """
    pkl_path = str(config.get("eval_bin_edges_pkl_path", "")).strip()
    if not pkl_path:
        return None
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"eval_bin_edges_pkl_path not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "bin_edges" not in data:
        raise KeyError(f"bin_edges not found in pkl: {pkl_path}")
    bin_edges = np.asarray(data["bin_edges"], dtype=np.float32)
    if len(bin_edges) != 4:
        raise ValueError(
            f"Only supports 3 bins now, expected len(bin_edges)=4, got {len(bin_edges)}: {bin_edges}"
        )
    low_max  = float(bin_edges[1])
    mid_min  = float(bin_edges[1])
    mid_max  = float(bin_edges[2])
    high_min = float(bin_edges[2])
    print("[PredictorEval] Loaded thresholds from pkl")
    print(f"[PredictorEval] pkl      : {pkl_path}")
    print(f"[PredictorEval] bin_edges: {bin_edges}")
    print(f"[PredictorEval] low<= {low_max:.6f}, mid=({mid_min:.6f}, {mid_max:.6f}], high> {high_min:.6f}")
    return low_max, mid_min, mid_max, high_min


# ---------------------------------------------------------------------------
# Predictor evaluator
# ---------------------------------------------------------------------------

class EcoliLSTMPredictorEval:
    """
    E. coli predictor evaluator:
    - load PyTorch LSTM predictor
    - predict normalized scores
    - inverse transform back to raw strength space
    """

    def __init__(self, config: Dict, device: torch.device):
        self.enabled = bool(config.get("use_predictor_eval", True))
        self.device  = device
        self.config  = config

        self.model_path       = str(config.get("ecoli_predictor_model_path", "")).strip()
        self.scaler_data_path = str(config.get("ecoli_predictor_data_path",  "")).strip()

        self.hidden_size  = int(config.get("ecoli_predictor_hidden_size",   256))
        self.dropout_rate = float(config.get("ecoli_predictor_dropout_rate", 0.2))
        self.lambda_l2    = float(config.get("ecoli_predictor_lambda_l2",    0.001))
        self.batch_size   = int(config.get("ecoli_predictor_batch_size",    256))

        thresholds = load_eval_thresholds_from_bin_edges(config)
        if thresholds is not None:
            self.low_max, self.mid_min, self.mid_max, self.high_min = thresholds
        else:
            self.low_max  = float(config.get("eval_low_max",  0.0))
            self.mid_min  = float(config.get("eval_mid_min",  0.0))
            self.mid_max  = float(config.get("eval_mid_max",  0.0))
            self.high_min = float(config.get("eval_high_min", 0.0))

        self.sequence_length = int(config.get("sequence_length", 165))

        self.model       = None
        self.scaler_mean = None
        self.scaler_std  = None

        if not self.enabled:
            print("[EcoliPredictorEval] Disabled")
            return

        missing = []
        if not self.model_path:
            missing.append("ecoli_predictor_model_path")
        if not self.scaler_data_path:
            missing.append("ecoli_predictor_data_path")

        if missing:
            print(f"[EcoliPredictorEval] Disabled: missing config keys: {missing}")
            self.enabled = False
            return

        if not os.path.exists(self.model_path):
            print(f"[EcoliPredictorEval] Disabled: model not found: {self.model_path}")
            self.enabled = False
            return

        if not os.path.exists(self.scaler_data_path):
            print(f"[EcoliPredictorEval] Disabled: scaler data not found: {self.scaler_data_path}")
            self.enabled = False
            return

        self._load_scaler_stats()
        self._build_and_load_model()

        print("[EcoliPredictorEval] Enabled")
        print(f"[EcoliPredictorEval] Model   : {self.model_path}")
        print(f"[EcoliPredictorEval] Data    : {self.scaler_data_path}")
        print(
            f"[EcoliPredictorEval] Thresholds: "
            f"low<={self.low_max}, mid=[{self.mid_min},{self.mid_max}], high>={self.high_min}"
        )

    def _load_scaler_stats(self):
        import pickle
    
        scaler_pkl = str(self.config.get("ecoli_predictor_scaler_path", "")).strip()
    
        # ✅ 优先加载训练时保存的 scaler
        if scaler_pkl and os.path.exists(scaler_pkl):
            with open(scaler_pkl, "rb") as f:
                scaler = pickle.load(f)
            self.scaler_mean = float(scaler.mean_[0])
            self.scaler_std  = float(scaler.scale_[0])
            print(f"[Scaler] Loaded from pkl : {scaler_pkl}")
            print(f"[Scaler] mean={self.scaler_mean:.4f}, std={self.scaler_std:.4f}")
            return
    
        # ❌ fallback：从 CSV 重新 fit（有尺度错位风险，仅兼容旧逻辑）
        print("[Scaler][WARN] ecoli_predictor_scaler_path not found, refitting from CSV")
        print(f"[Scaler][WARN] path tried: '{scaler_pkl}'")
    
        df = pd.read_csv(self.scaler_data_path)
        df.columns = [c.strip().lower() for c in df.columns]
    
        if "strength" not in df.columns:
            raise ValueError("ecoli_predictor_data_path must contain 'strength' column")
    
        strength = pd.to_numeric(df["strength"], errors="coerce").dropna().values.astype(np.float32)
        if len(strength) == 0:
            raise ValueError("No valid strength values found for scaler fitting")
    
        self.scaler_mean = float(np.mean(strength))
        self.scaler_std  = float(np.std(strength))
        if self.scaler_std <= 0:
            self.scaler_std = 1.0
    
        print(f"[Scaler] mean={self.scaler_mean:.4f}, std={self.scaler_std:.4f}")

    def _build_and_load_model(self):
        model = LSTMModel(
            input_size=4,
            hidden_size=self.hidden_size,
            output_size=1,
            dropout_rate=self.dropout_rate,
            lambda_l2=self.lambda_l2,
        )
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def _standardize_seq(self, s: str) -> str:
        s = str(s).upper().strip().replace("U", "T")
        return re.sub(r"[^ACGT]", "", s)

    def _seq_to_tensor(self, seq_list: List[str]) -> torch.Tensor:
        """Return: [B, L, 4]"""
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        arrs = []
        for s in seq_list:
            s = self._standardize_seq(s)
            if len(s) > self.sequence_length:
                s = s[:self.sequence_length]
            elif len(s) < self.sequence_length:
                s = s + "N" * (self.sequence_length - len(s))
            onehot = np.zeros((self.sequence_length, 4), dtype=np.float32)
            for i, ch in enumerate(s):
                if ch in mapping:
                    onehot[i, mapping[ch]] = 1.0
            arrs.append(onehot)
        x = np.stack(arrs, axis=0)  # [B, L, 4]
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def predict_raw(self, seq_list: List[str]) -> np.ndarray:
        """Return predictions in ORIGINAL E. coli strength space."""
        if (not self.enabled) or len(seq_list) == 0:
            return np.array([])

        x = self._seq_to_tensor(seq_list)
        preds = []
        with torch.no_grad():
            for i in range(0, x.shape[0], self.batch_size):
                xb     = x[i:i + self.batch_size]
                y_norm = self.model(xb).squeeze(-1).detach().cpu().numpy()
                preds.append(y_norm)

        preds_norm = np.concatenate(preds, axis=0)
        preds_raw  = preds_norm * self.scaler_std + self.scaler_mean
        return preds_raw.astype(np.float32)

    def evaluate_sequences(
            self,
            seq_list: List[str],
            target_bin_ids: Optional[np.ndarray] = None,
        ) -> Dict[str, float]:
            res: Dict[str, float] = {}
            preds_raw = self.predict_raw(seq_list)
            if preds_raw.size == 0:
                return res

            res["pred_mean"] = float(np.mean(preds_raw))
            res["pred_std"]  = float(np.std(preds_raw))
            res["pred_min"]  = float(np.min(preds_raw))
            res["pred_max"]  = float(np.max(preds_raw))

            if target_bin_ids is not None and len(target_bin_ids) == len(preds_raw):
                target_bin_ids = np.asarray(target_bin_ids, dtype=int)

                hit_mask = np.zeros(len(preds_raw), dtype=bool)
                dist     = np.zeros(len(preds_raw), dtype=np.float32)

                # 🚨 修正：显式对齐 1, 2, 3，并且将日志键名规范化
                low_mask = target_bin_ids == 1
                if low_mask.any():
                    low_hit = preds_raw[low_mask] <= self.low_max
                    hit_mask[low_mask] = low_hit
                    dist[low_mask]     = np.maximum(preds_raw[low_mask] - self.low_max, 0.0)
                    res["target_bin_1_pred_mean"] = float(np.mean(preds_raw[low_mask]))
                    res["target_bin_1_pred_std"]  = float(np.std(preds_raw[low_mask]))
                    res["low_hit_rate"]            = float(np.mean(low_hit))

                mid_mask = target_bin_ids == 2
                if mid_mask.any():
                    mid_vals = preds_raw[mid_mask]
                    mid_hit  = (mid_vals > self.mid_min) & (mid_vals <= self.mid_max)
                    hit_mask[mid_mask] = mid_hit
                    mid_dist = np.where(
                        mid_vals < self.mid_min, self.mid_min - mid_vals,
                        np.where(mid_vals > self.mid_max, mid_vals - self.mid_max, 0.0)
                    )
                    dist[mid_mask] = mid_dist
                    res["target_bin_2_pred_mean"] = float(np.mean(mid_vals))
                    res["target_bin_2_pred_std"]  = float(np.std(mid_vals))
                    res["mid_hit_rate"]            = float(np.mean(mid_hit))

                high_mask = target_bin_ids == 3
                if high_mask.any():
                    high_hit = preds_raw[high_mask] > self.high_min
                    hit_mask[high_mask] = high_hit
                    dist[high_mask]     = np.maximum(self.high_min - preds_raw[high_mask], 0.0)
                    res["target_bin_3_pred_mean"] = float(np.mean(preds_raw[high_mask]))
                    res["target_bin_3_pred_std"]  = float(np.std(preds_raw[high_mask]))
                    res["high_hit_rate"]           = float(np.mean(high_hit))

                res["condition_acc"] = float(np.mean(hit_mask))
                res["condition_mae"] = float(np.mean(dist))

            return res


# ---------------------------------------------------------------------------
# Yeast oracle predictor evaluator
# ---------------------------------------------------------------------------

class YeastOraclePredictorEval:
    """
    酵母菌 oracle 评估器（DeepSTARR-style TF1 模型，通过 subprocess 调用）。

    工作流程：
      1. 将生成序列写入临时 txt（纯序列，每行一条）
      2. 调用旧环境脚本 predict_yeast_strength.py
         脚本内部负责：加 flank → seq2feature → model.predict → scaler.inverse_transform
      3. 读取输出 txt（每行：seq\\tpred_strength）

    Flanking 序列（与 aux.py 保持一致）：
      left  = TGCATTTTTTTCACATC  (17 bp)
      right = GGTTACGGCTGTT       (13 bp)
      核心序列长度 80 bp → 加 flank 后 110 bp

    必需 config 键：
      yeast_python_bin   : 旧 TF 环境的 python 路径
      yeast_script_path  : predict_yeast_strength.py 的绝对路径

    可选 config 键：
      yeast_model_conditions : oracle model_conditions 目录名（默认 "defined_media"）
      eval_bin_edges_pkl_path: 分箱边界 pkl（与训练保持一致）
      eval_low_max / eval_mid_min / eval_mid_max / eval_high_min：手动阈值
    """

    # 与 aux.py 中完全一致的 flank 序列
    _LEFT_FLANK  = "TGCATTTTTTTCACATC"   # 17 bp
    _RIGHT_FLANK = "GGTTACGGCTGTT"        # 13 bp
    _CORE_LEN    = 80
    _FULL_LEN    = 110                     # 17 + 80 + 13

    def __init__(self, config: Dict, device: torch.device):
        self.enabled  = bool(config.get("use_predictor_eval", True))
        self.device   = device
        self.config   = config

        self.python_bin       = str(config.get("yeast_python_bin",       "")).strip()
        self.script_path      = str(config.get("yeast_script_path",      "")).strip()
        self.model_conditions = str(config.get("yeast_model_conditions", "defined_media")).strip()

        thresholds = load_eval_thresholds_from_bin_edges(config)
        if thresholds is not None:
            self.low_max, self.mid_min, self.mid_max, self.high_min = thresholds
        else:
            self.low_max  = float(config.get("eval_low_max",  0.0))
            self.mid_min  = float(config.get("eval_mid_min",  0.0))
            self.mid_max  = float(config.get("eval_mid_max",  0.0))
            self.high_min = float(config.get("eval_high_min", 0.0))

        if not self.enabled:
            print("[YeastOracleEval] Disabled")
            return

        missing = []
        if not self.python_bin:
            missing.append("yeast_python_bin")
        if not self.script_path:
            missing.append("yeast_script_path")

        if missing:
            print(f"[YeastOracleEval] Disabled: missing config keys: {missing}")
            self.enabled = False
            return

        if not os.path.exists(self.python_bin):
            print(f"[YeastOracleEval] Disabled: python not found: {self.python_bin}")
            self.enabled = False
            return

        if not os.path.exists(self.script_path):
            print(f"[YeastOracleEval] Disabled: script not found: {self.script_path}")
            self.enabled = False
            return

        print("[YeastOracleEval] Enabled")
        print(f"[YeastOracleEval] Python : {self.python_bin}")
        print(f"[YeastOracleEval] Script : {self.script_path}")
        print(f"[YeastOracleEval] Mode   : {self.model_conditions}")
        print(
            f"[YeastOracleEval] Thresholds: "
            f"low<={self.low_max}, mid=({self.mid_min},{self.mid_max}], high>{self.high_min}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _standardize_seq(self, s: str) -> str:
        """清理序列：大写、去非 ACGT 字符、截断或补 N 到核心长度。"""
        s = str(s).upper().strip().replace("U", "T")
        s = re.sub(r"[^ACGT]", "", s)
        if len(s) > self._CORE_LEN:
            s = s[:self._CORE_LEN]
        elif len(s) < self._CORE_LEN:
            s = s + "N" * (self._CORE_LEN - len(s))
        return s

    def _read_pred_txt(self, path: str) -> np.ndarray:
        """
        读取预测输出文件。
        格式：每行 <seq>\\t<pred_strength>  或  <pred_strength>（单列）
        """
        preds = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                # 优先取第 2 列，否则取第 1 列
                target = parts[1] if len(parts) >= 2 else parts[0]
                try:
                    preds.append(float(target))
                except ValueError:
                    continue   # 跳过 header 行
        return np.asarray(preds, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_raw(self, seq_list: List[str]) -> np.ndarray:
        """
        返回每条序列对应的 oracle 预测强度（inverse-scaled 原始空间）。
        脚本内部负责加 flank，此处只做序列清理。
        """
        if (not self.enabled) or len(seq_list) == 0:
            return np.array([])

        clean_seqs = [self._standardize_seq(s) for s in seq_list]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_txt  = os.path.join(tmpdir, "yeast_input.txt")
            output_txt = os.path.join(tmpdir, "yeast_output.txt")

            with open(input_txt, "w", encoding="utf-8") as f:
                for seq in clean_seqs:
                    f.write(seq + "\n")

            cmd = [
                self.python_bin,
                self.script_path,
                "--input_txt",        input_txt,
                "--model_conditions", self.model_conditions,
                "--output_txt",       output_txt,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print("[YeastOracleEval] subprocess failed")
                if result.stdout:
                    print("[YeastOracleEval][stdout]")
                    print(result.stdout)
                if result.stderr:
                    print("[YeastOracleEval][stderr]")
                    print(result.stderr)
                return np.array([])

            if not os.path.exists(output_txt):
                print("[YeastOracleEval] output_txt not found after subprocess")
                return np.array([])

            preds = self._read_pred_txt(output_txt)

        if len(preds) != len(clean_seqs):
            print(
                f"[YeastOracleEval][WARN] prediction count mismatch: "
                f"{len(preds)} preds vs {len(clean_seqs)} seqs"
            )

        return preds

    def evaluate_sequences(
            self,
            seq_list: List[str],
            target_bin_ids: Optional[np.ndarray] = None,
        ) -> Dict[str, float]:
            res: Dict[str, float] = {}
            preds_raw = self.predict_raw(seq_list)
            if preds_raw.size == 0:
                return res
    
            res["pred_mean"] = float(np.mean(preds_raw))
            res["pred_std"]  = float(np.std(preds_raw))
    
            if target_bin_ids is not None and len(target_bin_ids) == len(preds_raw):
                target_bin_ids = np.asarray(target_bin_ids, dtype=int)
    
                hit_mask = np.zeros(len(preds_raw), dtype=bool)
                dist     = np.zeros(len(preds_raw), dtype=np.float32)
    
                # 🚨 修正：将酵母菌的判定也同步调整为 1, 2, 3，消除物种间的分箱数据冲突
                low_mask = target_bin_ids == 1
                if low_mask.any():
                    low_hit = preds_raw[low_mask] <= self.low_max
                    hit_mask[low_mask] = low_hit
                    dist[low_mask]     = np.maximum(preds_raw[low_mask] - self.low_max, 0.0)
                    res["target_bin_1_pred_mean"] = float(np.mean(preds_raw[low_mask]))
                    res["low_hit_rate"]            = float(np.mean(low_hit))
    
                mid_mask = target_bin_ids == 2
                if mid_mask.any():
                    mid_vals = preds_raw[mid_mask]
                    mid_hit  = (mid_vals > self.mid_min) & (mid_vals <= self.mid_max)
                    hit_mask[mid_mask] = mid_hit
                    mid_dist = np.where(
                        mid_vals < self.mid_min, self.mid_min - mid_vals,
                        np.where(mid_vals > self.mid_max, mid_vals - self.mid_max, 0.0)
                    )
                    dist[mid_mask] = mid_dist
                    res["target_bin_2_pred_mean"] = float(np.mean(mid_vals))
                    res["mid_hit_rate"]            = float(np.mean(mid_hit))
    
                high_mask = target_bin_ids == 3
                if high_mask.any():
                    high_hit = preds_raw[high_mask] > self.high_min
                    hit_mask[high_mask] = high_hit
                    dist[high_mask]     = np.maximum(self.high_min - preds_raw[high_mask], 0.0)
                    res["target_bin_3_pred_mean"] = float(np.mean(preds_raw[high_mask]))
                    res["high_hit_rate"]           = float(np.mean(high_hit))
    
                res["condition_acc"] = float(np.mean(hit_mask))
                res["condition_mae"] = float(np.mean(dist))
    
            return res

def build_predictor_eval(config: Dict, device: torch.device):
    predictor_type = str(config.get("predictor_eval_type", "ecoli_lstm")).strip().lower()

    if predictor_type in {"ecoli_lstm", "ecoli_predictor"}:
        return EcoliLSTMPredictorEval(config, device)

    if predictor_type in {"yeast_oracle", "yeast"}:
        return YeastOraclePredictorEval(config, device)

    print(f"[PredictorEval] Unknown predictor_eval_type={predictor_type}, disabled.")
    dummy = type("DummyEval", (), {"enabled": False})()
    return dummy
# ---------------------------------------------------------------------------
# Sequence-level utility functions
# ---------------------------------------------------------------------------

def get_kmer_counts(seq_list: List[str], alphabet: str = "ACGT", k: int = 3) -> np.ndarray:
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    kmer_map  = {km: i for i, km in enumerate(all_kmers)}
    counts    = np.zeros(len(all_kmers), dtype=np.float64)

    for s in seq_list:
        s_clean = s.replace("P", "")
        for i in range(len(s_clean) - k + 1):
            sub = s_clean[i:i + k]
            if sub in kmer_map:
                counts[kmer_map[sub]] += 1

    total = counts.sum()
    return counts / (total + 1e-10)


def compute_batch_edit_distance(
    real_seqs: List[str], fake_seqs: List[str], max_pairs: int = 64
) -> float:
    if len(real_seqs) == 0 or len(fake_seqs) == 0:
        return 0.0

    n = min(len(real_seqs), len(fake_seqs), max_pairs)
    if n <= 0:
        return 0.0

    real_idx = np.random.choice(len(real_seqs), n, replace=False)
    fake_idx = np.random.choice(len(fake_seqs), n, replace=False)

    dists = [
        editdistance.eval(
            real_seqs[ir].replace("P", ""),
            fake_seqs[jf].replace("P", ""),
        )
        for ir, jf in zip(real_idx, fake_idx)
    ]
    return float(np.mean(dists)) if dists else 0.0


def gc_content(seq_list: List[str]) -> float:
    all_str = "".join(seq_list).replace("P", "")
    if len(all_str) == 0:
        return 0.0
    return float((all_str.count("G") + all_str.count("C")) / len(all_str))


def _decode_dataset_sequence(x: torch.Tensor, nucleotides: List[str]) -> str:
    arr = x.squeeze(0).cpu().numpy() if x.ndim == 3 else x.cpu().numpy()
    arr = (arr > 0).astype(np.float32)
    return convert_to_seq(arr, nucleotides)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_gc_bin_pool_by_condition_from_dataset(
    dataset,
    condition_id_list: List[int],
) -> Optional[Dict[int, np.ndarray]]:
    """
    从 Dataset 中构建每个 condition 对应的真实 GC bin 分布。
    gc_bin=0 保留给 unconditional，不用于真实条件采样。
    """
    if not hasattr(dataset, "labels") or not hasattr(dataset, "gc_bin_ids"):
        print("[Eval][WARN] dataset does not contain labels/gc_bin_ids. GC-conditional eval sampling disabled.")
        return None

    labels_np = (
        dataset.labels.detach().cpu().numpy()
        if torch.is_tensor(dataset.labels)
        else np.asarray(dataset.labels)
    )
    gc_bins_np = np.asarray(dataset.gc_bin_ids)

    pool_by_condition = {}
    for cid in condition_id_list:
        cid  = int(cid)
        pool = gc_bins_np[labels_np == cid]
        pool = pool[pool > 0]
        pool_by_condition[cid] = pool

    return pool_by_condition


# ---------------------------------------------------------------------------
# Sampling & validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_sequences_by_condition(
    model,
    condition_id_list: List[int],
    numeric_to_tag_dict: Dict[int, str],
    device: torch.device,
    num_samples_per_condition: int = 200,
    sample_bs: int = 50,
    cond_weight: float = 1.0,
    sequence_length: int = 80,
    nucleotides: Optional[List[str]] = None,
    gc_bin_pool_by_condition: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, List[str]]:
    if nucleotides is None:
        nucleotides = ["A", "C", "G", "T"]

    fake_grouped: Dict[str, List[str]] = {}
    model.eval()

    for cond_id in condition_id_list:
        cond_id           = int(cond_id)
        tag               = numeric_to_tag_dict[cond_id]
        fake_grouped[tag] = []
        remaining         = num_samples_per_condition

        while remaining > 0:
            cur_bs  = min(sample_bs, remaining)
            classes = torch.full((cur_bs,), cond_id, dtype=torch.long, device=device)

            gc_bin = None
            if gc_bin_pool_by_condition is not None:
                if cond_id not in gc_bin_pool_by_condition:
                    raise ValueError(
                        f"gc_bin_pool_by_condition missing condition_id={cond_id}. "
                        f"Available keys={list(gc_bin_pool_by_condition.keys())}"
                    )
                pool = np.asarray(gc_bin_pool_by_condition[cond_id], dtype=np.int64)
                pool = pool[pool > 0]
                if len(pool) == 0:
                    raise ValueError(
                        f"condition_id={cond_id} has empty GC bin pool. "
                        "Check dataloader gc_bin_ids."
                    )
                gc_bin = torch.tensor(
                    np.random.choice(pool, size=cur_bs, replace=True),
                    dtype=torch.long,
                    device=device,
                )

            sampled_images = model.sample(
                classes=classes,
                gc_bin=gc_bin,
                shape=(cur_bs, 4, sequence_length),
                cond_weight=cond_weight,
            )

            for step in sampled_images[-1]:
                fake_grouped[tag].append(convert_to_seq(step, nucleotides))

            remaining -= cur_bs

    return fake_grouped


@torch.no_grad()
def collect_real_sequences_by_condition(
    val_data,
    numeric_to_tag_dict: Dict[int, str],
    max_real_per_condition: Optional[int] = None,
    nucleotides: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if nucleotides is None:
        nucleotides = ["A", "C", "G", "T"]

    real_grouped:  Dict[str, List[str]] = {}
    count_grouped: Dict[str, int]       = {}

    for i in range(len(val_data)):
        item   = val_data[i]
        x, y   = item[0], item[1]
        cond_id = int(y.item())
        tag     = numeric_to_tag_dict[cond_id]

        if tag not in real_grouped:
            real_grouped[tag]  = []
            count_grouped[tag] = 0

        if max_real_per_condition is not None and count_grouped[tag] >= max_real_per_condition:
            continue

        real_grouped[tag].append(_decode_dataset_sequence(x, nucleotides))
        count_grouped[tag] += 1

    return real_grouped


@torch.no_grad()
def validate_diffusion_model(
    model,
    data: tuple,
    device: torch.device,
    config: Dict,
    predictor_eval = None,
    num_samples_per_condition: int = 200,
    sample_bs: int = 50,
    cond_weight: float = 1.0,
    sequence_length: int = 80,
    max_real_per_condition: Optional[int] = 500,
    alphabet: str = "ACGT",
) -> Dict[str, float]:
    train_data, val_data, condition_id_list, numeric_to_tag_dict = data

    # 1) 物种与序列长度自动对齐与校验
    sequence_length = int(config.get("sequence_length", sequence_length))
    eval_type = str(config.get("predictor_eval_type", "ecoli_lstm")).strip().lower()

    if eval_type in {"ecoli_lstm", "ecoli_predictor"}:
        if sequence_length != 165: sequence_length = 165
    elif eval_type in {"yeast_oracle", "yeast"}:
        if sequence_length != 80: sequence_length = 80

    nucleotides = list(alphabet.replace("P", ""))
    eval_res: Dict[str, float] = {}

    # 2) 收集真实序列与采样
    real_grouped = collect_real_sequences_by_condition(val_data, numeric_to_tag_dict, max_real_per_condition, nucleotides)
    gc_bin_pool_by_condition = build_gc_bin_pool_by_condition_from_dataset(val_data, condition_id_list)

    fake_grouped = sample_sequences_by_condition(
        model=model, condition_id_list=condition_id_list, numeric_to_tag_dict=numeric_to_tag_dict,
        device=device, num_samples_per_condition=num_samples_per_condition, sample_bs=sample_bs,
        cond_weight=cond_weight, sequence_length=sequence_length, nucleotides=nucleotides, gc_bin_pool_by_condition=gc_bin_pool_by_condition
    )

    all_real: List[str] = []
    all_fake: List[str] = []
    all_fake_target_bins: List[int] = []

    for cond_id in condition_id_list:
        tag = numeric_to_tag_dict[int(cond_id)]
        real_seqs = real_grouped.get(tag, [])
        fake_seqs = fake_grouped.get(tag, [])

        all_real.extend(real_seqs)
        all_fake.extend(fake_seqs)
        
        # 🚨 核心修正：不管是大肠杆菌还是酵母菌，外部统一保持原生的 1, 2, 3 标签传递给下层的评估器
        all_fake_target_bins.extend([int(cond_id)] * len(fake_seqs))

        eval_res[f"{tag}_gc_real"] = gc_content(real_seqs)
        eval_res[f"{tag}_gc_fake"] = gc_content(fake_seqs)

    # 3) 计算多聚体相关性与综合指标
    for k in [3, 4, 5]:
        r_counts = get_kmer_counts(all_real, alphabet=alphabet, k=k)
        f_counts = get_kmer_counts(all_fake, alphabet=alphabet, k=k)
        corr = float(np.corrcoef(r_counts, f_counts)[0, 1]) if not (np.all(r_counts==0) or np.all(f_counts==0)) else 0.0
        eval_res[f"{k}mer_R"] = corr

    eval_res["avg_corr"] = float(np.mean([eval_res["3mer_R"], eval_res["4mer_R"], eval_res["5mer_R"]]))
    eval_res["gc_real"] = gc_content(all_real)
    eval_res["gc_fake"] = gc_content(all_fake)
    eval_res["edit_dist"] = compute_batch_edit_distance(all_real, all_fake)

    # 4) 运行评估
    if predictor_eval is not None and predictor_eval.enabled:
        pred_metrics = predictor_eval.evaluate_sequences(all_fake, np.array(all_fake_target_bins))
        eval_res.update(pred_metrics)
        
        acc = eval_res.get("condition_acc", 0.0)
        eval_res["composite_score"] = float(0.5 * eval_res["3mer_R"] + 0.5 * acc)

    return eval_res