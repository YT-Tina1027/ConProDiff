import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 压制大部分 TF C++ 日志
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import gc
import argparse
import tensorflow as tf
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from aux import load_model, evaluate_model


def read_sequences_from_txt(txt_path):
    seqs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            seq = line.strip().upper()
            if seq:
                seqs.append(seq)
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_txt",
        type=str,
        required=True,
        help="Path to txt file containing one sequence per line",
    )
    parser.add_argument(
        "--model_conditions",
        type=str,
        default="complex_media",
        choices=["complex_media", "defined_media"],
        help="Which oracle model to use",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        required=True,
        help="Path to save predictions",
    )
    args = parser.parse_args()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

    seqs = read_sequences_from_txt(args.input_txt)
    print(f"Loaded {len(seqs)} sequences")

    fitness_function_graph = tf.Graph()
    with fitness_function_graph.as_default():
        model, scaler, batch_size = load_model(args.model_conditions)

    y_pred = evaluate_model(
        seqs,
        model,
        scaler,
        batch_size,
        fitness_function_graph
    )

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for seq, pred in zip(seqs, y_pred):
            f.write(f"{seq}\t{float(pred):.6f}\n")

    print(f"Saved predictions to: {args.output_txt}")


if __name__ == "__main__":
    main()