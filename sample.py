import hydra
import torch
import torch.nn as nn
import numpy as np

from omegaconf import DictConfig, OmegaConf
from dnadiffusion.utils.sample_util import create_sample


def build_gc_bin_pool_by_condition(
    train_data,
    condition_id_list,
    numeric_to_tag_dict,
):
    """
    从训练集构建每个 strength condition 对应的真实 GC bin 分布。

    采样时：
    - low  从真实 low 的 GC bin 分布中抽样
    - mid  从真实 mid 的 GC bin 分布中抽样
    - high 从真实 high 的 GC bin 分布中抽样

    返回:
        dict[int, np.ndarray]
    """
    if not hasattr(train_data, "labels") or not hasattr(train_data, "gc_bin_ids"):
        print("[WARN] train_data does not contain labels/gc_bin_ids. GC-conditional sampling disabled.")
        return None

    labels_np = (
        train_data.labels.detach().cpu().numpy()
        if torch.is_tensor(train_data.labels)
        else np.asarray(train_data.labels)
    )

    gc_bins_np = np.asarray(train_data.gc_bin_ids)

    gc_bin_pool_by_condition = {}

    for cid in condition_id_list:
        cid = int(cid)

        pool = gc_bins_np[labels_np == cid]

        # 0 是 unconditional GC，不用于真实条件采样
        pool = pool[pool > 0]

        gc_bin_pool_by_condition[cid] = pool

        if len(pool) > 0:
            print(
                f"[GC sampling pool] condition={cid} "
                f"({numeric_to_tag_dict.get(cid, cid)}): "
                f"n={len(pool)}, min={pool.min()}, max={pool.max()}"
            )
        else:
            print(
                f"[GC sampling pool] condition={cid} "
                f"({numeric_to_tag_dict.get(cid, cid)}): n=0"
            )

    return gc_bin_pool_by_condition


def sample(
    data: tuple,
    model: nn.Module,
    checkpoint_path: str,
    sample_batch_size: int,
    number_of_samples: int,
    guidance_scale: float,
    sequence_length: int,
    output_dir: str,
    use_gc_condition: bool = True,
) -> None:
    print("Loading checkpoint...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint_dict, dict) and "model" in checkpoint_dict:
        model.model.load_state_dict(checkpoint_dict["model"])
        print(f"Loaded checkpoint dict from: {checkpoint_path}")
    else:
        model.model.load_state_dict(checkpoint_dict)
        print(f"Loaded raw state_dict from: {checkpoint_path}")

    model = model.to(device)
    model.eval()
    print(f"Model sent to {device}")

    train_data, val_data, condition_id_list, numeric_to_tag_dict = data

    print(f"Found conditions: {[numeric_to_tag_dict[i] for i in condition_id_list]}")
    print("condition_id_list =", condition_id_list)
    print("numeric_to_tag_dict =", numeric_to_tag_dict)

    if hasattr(train_data, "num_gc_bins"):
        print("[Sample] train_data.num_gc_bins =", train_data.num_gc_bins)

    if hasattr(train_data, "gc_bin_width"):
        print("[Sample] train_data.gc_bin_width =", train_data.gc_bin_width)

    # =====================================================
    # 只在 use_gc_condition=True 时构建 GC pool
    # =====================================================
    if use_gc_condition:
        gc_bin_pool_by_condition = build_gc_bin_pool_by_condition(
            train_data=train_data,
            condition_id_list=condition_id_list,
            numeric_to_tag_dict=numeric_to_tag_dict,
        )
    else:
        gc_bin_pool_by_condition = None
        print("[Sample] use_gc_condition=False, GC-conditional sampling disabled.")

    for cond_id in condition_id_list:
        cond_id = int(cond_id)
        tag = numeric_to_tag_dict[cond_id]
        print(f"Generating {number_of_samples} samples for condition {tag}")

        create_sample(
            model,
            condition_id=cond_id,
            sample_bs=sample_batch_size,
            conditional_numeric_to_tag=numeric_to_tag_dict,
            number_of_samples=number_of_samples,
            cond_weight_to_metric=guidance_scale,
            save_timesteps=False,
            save_dataframe=True,
            generate_attention_maps=False,
            sequence_length=sequence_length,
            output_dir=output_dir,
            gc_bin_pool_by_condition=gc_bin_pool_by_condition,  # None → 不传 gc_bin
        )


@hydra.main(config_path="configs", config_name="sample", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    use_gc = cfg.model.use_gc_condition

    data = hydra.utils.instantiate(cfg.data)
    train_data = data[0]

    model = hydra.utils.instantiate(
        cfg.model,
        num_gc_bins=train_data.num_gc_bins if use_gc else None,  # ← 新增
    )

    diffusion = hydra.utils.instantiate(cfg.diffusion, model=model)

    sample(
        data=data,
        model=diffusion,
        checkpoint_path=cfg.sample.checkpoint_path,
        sample_batch_size=cfg.sample.sample_batch_size,
        number_of_samples=cfg.sample.number_of_samples,
        guidance_scale=cfg.sample.guidance_scale,
        sequence_length=cfg.data.sequence_length,
        output_dir=cfg.sample.output_dir,
        use_gc_condition=use_gc,
    )

if __name__ == "__main__":
    main()