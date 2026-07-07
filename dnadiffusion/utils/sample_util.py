import os
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dnadiffusion.utils.utils import convert_to_seq


def sample_gc_bins_for_condition(
    condition_id: int,
    current_bs: int,
    gc_bin_pool_by_condition: dict | None = None,
    device=None,
):
    """
    根据目标 strength condition，从对应真实 condition 的 GC bin 分布中抽样。

    参数：
    - condition_id:
        strength 条件编号，例如 1/2/3
    - current_bs:
        当前 batch size
    - gc_bin_pool_by_condition:
        dict[int, array-like]
        例如：
        {
            1: array([...]),  # real low 的 gc_bin
            2: array([...]),  # real mid 的 gc_bin
            3: array([...]),  # real high 的 gc_bin
        }

    返回：
    - gc_bin_tensor: LongTensor, shape [current_bs]
    """
    if gc_bin_pool_by_condition is None:
        return None

    if condition_id not in gc_bin_pool_by_condition:
        raise ValueError(
            f"gc_bin_pool_by_condition 中没有 condition_id={condition_id}。"
            f"已有 keys={list(gc_bin_pool_by_condition.keys())}"
        )

    pool = np.asarray(gc_bin_pool_by_condition[condition_id], dtype=np.int64)

    if len(pool) == 0:
        raise ValueError(f"condition_id={condition_id} 对应的 GC bin pool 为空。")

    # 注意：真实 GC bin 应该从 1 开始，0 只给 unconditional
    pool = pool[pool > 0]

    if len(pool) == 0:
        raise ValueError(
            f"condition_id={condition_id} 的 GC bin pool 里没有 >0 的真实 GC bin。"
            "请检查 dataloader.py 是否已经把真实 gc_bin 设置为从 1 开始。"
        )

    sampled_gc_bins = np.random.choice(
        pool,
        size=current_bs,
        replace=True,
    )

    return torch.tensor(
        sampled_gc_bins,
        dtype=torch.long,
        device=device,
    )


def create_sample(
    model: torch.nn.Module,
    condition_id: int,
    sample_bs: int,
    conditional_numeric_to_tag: dict,
    number_of_samples: int = 1000,
    cond_weight_to_metric: float = 0.0,
    save_timesteps: bool = False,
    save_dataframe: bool = False,
    generate_attention_maps: bool = False,
    sequence_length: int = 165,
    output_dir: str = "data/outputs",

    # ==============================
    # 新增：GC 条件采样相关参数
    # ==============================
    gc_bin_pool_by_condition: dict | None = None,
    fixed_gc_bin: int | None = None,
) -> None:
    """
    通用 conditional diffusion 采样函数（1D 版本）。

    参数说明：
    - condition_id:
        strength 条件编号，例如 1/2/3，对应 low/mid/high

    - sample_bs:
        每次采样 batch size

    - conditional_numeric_to_tag:
        条件 id -> 条件名字

    - number_of_samples:
        总采样数

    - sequence_length:
        序列长度

    - gc_bin_pool_by_condition:
        如果提供，则每次采样时从对应 condition 的真实 GC bin 分布中抽样。
        推荐使用这个方式。

    - fixed_gc_bin:
        如果提供，则所有样本都使用同一个 GC bin。
        这个主要用于调试，不推荐正式实验使用。

    注意：
    - 如果 gc_bin_pool_by_condition 和 fixed_gc_bin 都是 None，
      那么采样时不传 gc_bin，等价于只按 strength 条件采样。
    """
    os.makedirs(output_dir, exist_ok=True)

    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []

    num_batches = math.ceil(number_of_samples / sample_bs)

    for batch_idx in tqdm(range(num_batches)):
        current_bs = min(sample_bs, number_of_samples - batch_idx * sample_bs)

        if current_bs <= 0:
            break

        classes = torch.full(
            (current_bs,),
            condition_id,
            dtype=torch.long,
            device=model.device,
        )

        # =====================================================
        # 新增：构造 GC 条件
        # =====================================================
        if fixed_gc_bin is not None:
            if fixed_gc_bin <= 0:
                raise ValueError(
                    "fixed_gc_bin 应该是 >0 的真实 GC bin。"
                    "gc_bin=0 只用于 unconditional，不建议采样时固定为 0。"
                )

            gc_bin = torch.full(
                (current_bs,),
                int(fixed_gc_bin),
                dtype=torch.long,
                device=model.device,
            )

        else:
            gc_bin = sample_gc_bins_for_condition(
                condition_id=condition_id,
                current_bs=current_bs,
                gc_bin_pool_by_condition=gc_bin_pool_by_condition,
                device=model.device,
            )

        if generate_attention_maps:
            sampled_images, cross_att_values = model.sample_cross(
                classes=classes,
                gc_bin=gc_bin,
                shape=(current_bs, 4, sequence_length),
                cond_weight=cond_weight_to_metric,
            )

            np.save(
                os.path.join(
                    output_dir,
                    f"cross_att_values_{conditional_numeric_to_tag[condition_id]}.npy"
                ),
                cross_att_values,
            )

        else:
            sampled_images = model.sample(
                classes=classes,
                gc_bin=gc_bin,
                shape=(current_bs, 4, sequence_length),
                cond_weight=cond_weight_to_metric,
            )

        final_step = sampled_images[-1]

        if save_timesteps:
            seqs_to_df = {}

            for step_idx, step in enumerate(sampled_images):
                seqs_to_df[step_idx] = [
                    convert_to_seq(x, nucleotides)
                    for x in step
                ]

            final_sequences.append(pd.DataFrame(seqs_to_df))

        elif save_dataframe:
            for step in final_step:
                final_sequences.append(convert_to_seq(step, nucleotides))

        else:
            for n_b, x in enumerate(final_step):
                seq_final = f">seq_test_{batch_idx}_{n_b}\n" + "".join(
                    [
                        nucleotides[s]
                        for s in np.argmax(
                            x.reshape(4, sequence_length),
                            axis=0
                        )
                    ]
                )

                final_sequences.append(seq_final)

    out_path = os.path.join(
        output_dir,
        f"{conditional_numeric_to_tag[condition_id]}.txt"
    )

    if save_timesteps:
        pd.concat(final_sequences, ignore_index=True).to_csv(
            out_path,
            header=True,
            sep="\t",
            index=False,
        )
        return

    if save_dataframe:
        with open(out_path, "w") as f:
            f.write("\n".join(final_sequences))
        return

    with open(out_path, "w") as f:
        f.write("\n".join(final_sequences))