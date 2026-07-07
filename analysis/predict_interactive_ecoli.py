"""
predict_interactive.py
=======================
在终端输入 165bp DNA 序列，实时输出预测表达强度。

用法：
    python predict_interactive.py
    python predict_interactive.py --model /path/to/model.pth  # 自定义模型路径

支持：
    - 单条序列交互输入
    - 批量粘贴多条序列（空行分隔，输入 END 结束）
    - 大小写不敏感，自动过滤非 ATCG 字符
    - 输入 q / quit / exit 退出
"""

import os
import sys
import gc
import re
import pickle
import argparse
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# =========================================================
# ★ 默认路径（可通过命令行参数覆盖）★
# =========================================================

DEFAULT_MODEL_PATH  = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/LSTMModel_EC_best.pth"
DEFAULT_SCALER_PATH = "/home/yt/Code/DNA-Diffusion/Predictor/results/model_EC/scaler_EC.pkl"
DEFAULT_BIO_ROOT    = "/home/yt/Code/DNA-Diffusion"

REAL_CSV   = "/home/yt/Code/DNA-Diffusion/Data/ecoli_exp_all.csv"  # 用于显示分位数参考
SEQ_LEN    = 165

# =========================================================
# 工具函数
# =========================================================

def standardize(seq: str) -> str:
    seq = str(seq).upper().strip().replace("U", "T")
    return "".join(ch for ch in seq if ch in "ATCG")

def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0

def load_bin_edges(csv_path: str):
    """从训练集计算 low/mid/high 的分位数边界，用于结果解读。"""
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "strength" not in df.columns:
            return None, None
        vals = pd.to_numeric(df["strength"], errors="coerce").dropna().values
        return float(np.percentile(vals, 33.333)), float(np.percentile(vals, 66.667))
    except Exception:
        return None, None

def strength_label(pred: float, q33, q67) -> str:
    if q33 is None or q67 is None:
        return ""
    if pred < q33:
        return "  → LOW  组"
    elif pred > q67:
        return "  → HIGH 组"
    else:
        return "  → MID  组"

# =========================================================
# 模型加载（只加载一次，整个会话复用）
# =========================================================

def load_model(model_path, scaler_path, bio_root):
    import torch
    if bio_root not in sys.path:
        sys.path.insert(0, bio_root)
    from Predictor.predictor_models import LSTMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = LSTMModel(4, 256, 1, 0.2, 0.001)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, scaler, device

# =========================================================
# 编码 & 预测
# =========================================================

def encode(seq: str) -> "torch.Tensor":
    import torch
    import numpy as np
    seq = re.sub("[^acgt]", "z", seq.lower().replace("u", "t"))
    mapping = {"a": 0, "c": 1, "g": 2, "t": 3}
    x = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    for i, ch in enumerate(seq[:SEQ_LEN]):
        if ch in mapping:
            x[i, mapping[ch]] = 1.0
    return torch.tensor(x)

def predict_batch(seqs, model, scaler, device):
    import torch
    tensors = [encode(s) for s in seqs]
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        y_norm = model(x).detach().cpu().numpy().reshape(-1, 1)
        y_raw  = scaler.inverse_transform(y_norm).reshape(-1)
    return y_raw

# =========================================================
# 输入验证
# =========================================================

def validate(raw: str):
    """
    返回 (clean_seq, error_msg)
    error_msg 为 None 表示合法
    """
    seq = standardize(raw)
    if len(seq) == 0:
        return None, "输入为空或不含有效碱基（ATCG）"
    if len(seq) != SEQ_LEN:
        return None, (f"长度错误：输入 {len(raw)} 字符，"
                      f"有效碱基 {len(seq)} 个，需要恰好 {SEQ_LEN} bp")
    return seq, None

# =========================================================
# 结果打印
# =========================================================

DIVIDER = "─" * 52

def print_result(seq, pred, idx, q33, q67):
    gc = gc_content(seq)
    label = strength_label(pred, q33, q67)
    print(f"\n  序列 #{idx}")
    print(f"  {DIVIDER}")
    print(f"  前30bp  : {seq[:30]}...")
    print(f"  GC含量  : {gc*100:.1f}%")
    print(f"  预测强度: {pred:+.4f}{label}")
    print(f"  {DIVIDER}")

# =========================================================
# 主交互循环
# =========================================================

def run(model, scaler, device, q33, q67):
    QUIT_CMDS = {"q", "quit", "exit", "退出"}

    print(f"\n{'='*52}")
    print(f"  E. coli 启动子表达强度预测器")
    print(f"  序列长度：{SEQ_LEN} bp")
    if q33 is not None:
        print(f"  参考边界：Low < {q33:.3f} ≤ Mid ≤ {q67:.3f} < High")
    print(f"{'='*52}")
    print("  输入模式：")
    print("    单条  → 直接粘贴序列后回车")
    print("    多条  → 每行一条，最后输入 END 回车")
    print("    退出  → 输入 q / quit / exit")
    print(f"{'='*52}\n")

    counter = 0

    while True:
        try:
            line = input("请输入序列 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not line:
            continue

        if line.lower() in QUIT_CMDS:
            print("已退出。")
            break

        # ── 多条模式：首行不是完整序列，或用户输入 BATCH ──
        if line.upper() == "BATCH":
            print("  多条模式：每行一条序列，输入 END 结束")
            raw_lines = []
            while True:
                try:
                    l = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if l.upper() == "END":
                    break
                if l:
                    raw_lines.append(l)

            if not raw_lines:
                print("  未输入任何序列。\n")
                continue

            valid_seqs, errors = [], []
            for i, raw in enumerate(raw_lines, 1):
                seq, err = validate(raw)
                if err:
                    errors.append((i, raw[:20], err))
                else:
                    valid_seqs.append((i, seq))

            if errors:
                print(f"\n  以下 {len(errors)} 条序列有误，已跳过：")
                for i, preview, err in errors:
                    print(f"    #{i} ({preview}...): {err}")

            if valid_seqs:
                seqs_only = [s for _, s in valid_seqs]
                preds = predict_batch(seqs_only, model, scaler, device)
                for (idx, seq), pred in zip(valid_seqs, preds):
                    counter += 1
                    print_result(seq, pred, idx, q33, q67)
            print()

        else:
            # ── 单条模式 ──
            seq, err = validate(line)
            if err:
                print(f"  [错误] {err}\n")
                continue

            counter += 1
            preds = predict_batch([seq], model, scaler, device)
            print_result(seq, preds[0], counter, q33, q67)
            print()

# =========================================================
# 入口
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="E. coli 启动子表达强度交互预测")
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH,  help="模型 .pth 路径")
    parser.add_argument("--scaler", default=DEFAULT_SCALER_PATH, help="Scaler .pkl 路径")
    parser.add_argument("--root",   default=DEFAULT_BIO_ROOT,    help="项目根目录（sys.path）")
    args = parser.parse_args()

    # 检查文件存在
    for path, name in [(args.model, "模型"), (args.scaler, "Scaler")]:
        if not os.path.exists(path):
            print(f"[错误] {name}文件不存在: {path}")
            sys.exit(1)

    print("正在加载模型...")
    model, scaler, device = load_model(args.model, args.scaler, args.root)
    print(f"模型已加载（{device}）| Scaler mean={scaler.mean_[0]:.4f}")

    print("正在读取训练集分位数边界...")
    q33, q67 = load_bin_edges(REAL_CSV)

    run(model, scaler, device, q33, q67)

if __name__ == "__main__":
    main()