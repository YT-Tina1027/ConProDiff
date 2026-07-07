import os
import numpy as np

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from dnadiffusion.data.dataloader import get_dataloader
from dnadiffusion.utils.sample_util import create_sample
from dnadiffusion.utils.train_util import distributed_setup, init_wandb, train_step, val_step
from dnadiffusion.utils.evaluator_diffusion import (
    build_predictor_eval,
    validate_diffusion_model,
    build_gc_bin_pool_by_condition_from_dataset, 
)

# =========================================================
# 全局保存路径：以后只改这里
# =========================================================
CHECKPOINT_DIR = "/home/yt/Code/DNA-Diffusion/result/checkpoints_yeast_nogc"
SAMPLE_OUTPUT_DIR = "/home/yt/Code/DNA-Diffusion/result/sample_yeast_nogc"


def compute_eval_score(eval_metrics: dict, strength_range: float = 17.0) -> float:
    """
    综合评估分数，越大越好。
    """
    if not eval_metrics:
        return float("-inf")

    condition_acc = float(eval_metrics.get("condition_acc", 0.0))
    avg_corr = float(eval_metrics.get("avg_corr", 0.0))
    condition_mae = float(eval_metrics.get("condition_mae", strength_range))
    gc_real = float(eval_metrics.get("gc_real", 0.0))
    gc_fake = float(eval_metrics.get("gc_fake", 0.0))

    low_mean = float(eval_metrics.get("target_bin_1_pred_mean", 0.0))
    mid_mean = float(eval_metrics.get("target_bin_2_pred_mean", 0.0))
    high_mean = float(eval_metrics.get("target_bin_3_pred_mean", 0.0))

    condition_mae_norm = condition_mae / strength_range
    gc_gap = abs(gc_fake - gc_real)

    sep_low_mid = max(mid_mean - low_mean, 0.0)
    sep_mid_high = max(high_mean - mid_mean, 0.0)
    bin_separation = (sep_low_mid + sep_mid_high) / 2.0
    bin_separation_norm = bin_separation / strength_range

    score = (
        0.40 * condition_acc
        + 0.25 * avg_corr
        - 0.15 * condition_mae_norm
        - 0.10 * gc_gap
        + 0.10 * bin_separation_norm
    )

    return float(score)


def get_current_state_dicts(model, optimizer, distributed: bool):
    """
    统一获取当前模型和优化器 state_dict。
    """
    if distributed:
        model_state, optimizer_state = get_state_dict(model.module.model, optimizer)
    else:
        model_state = model.model.state_dict()
        optimizer_state = optimizer.state_dict()

    return model_state, optimizer_state


def train(
    distributed: bool,
    precision: str,
    num_workers: int,
    pin_memory: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: tuple,
    batch_size: int,
    sample_batch_size: int,
    log_step: int,
    num_epochs: int,
    min_epochs: int,
    patience: int,
    sample_epoch: int,
    number_of_samples: int,
    use_wandb: bool,
    predictor_eval=None,
    eval_every_n_epochs: int = 20,
    eval_num_samples_per_condition: int = 100,
    sequence_length: int = 80,
    use_gc_condition: bool = True,  
    cond_weight: float = 1.0, 
    random_seed: int = 42,
    global_cfg_dict: dict = None,  # 💡 新增参数：接收完整配置，避免传递空字典给验证函数
) -> None:
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank, device, local_batch_size = distributed_setup(batch_size)
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

        model = DDP(model.to(device), device_ids=[rank])
        rank_0 = rank == 0
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        rank_0 = True
        local_batch_size = batch_size

    train_data, val_data, condition_id_list, numeric_to_tag_dict = data

    print("condition_id_list =", condition_id_list)
    print("numeric_to_tag_dict =", numeric_to_tag_dict)

    if hasattr(train_data, "num_gc_bins"):
        print("[Train] train_data.num_gc_bins =", train_data.num_gc_bins)

    if hasattr(train_data, "gc_bin_width"):
        print("[Train] train_data.gc_bin_width =", train_data.gc_bin_width)

    print("[Save] CHECKPOINT_DIR =", CHECKPOINT_DIR)
    print("[Save] SAMPLE_OUTPUT_DIR =", SAMPLE_OUTPUT_DIR)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

    if use_gc_condition:
        gc_bin_pool_by_condition = build_gc_bin_pool_by_condition_from_dataset(
            dataset=train_data,
            condition_id_list=condition_id_list,
        )
        if gc_bin_pool_by_condition is not None:
            for cid, pool in gc_bin_pool_by_condition.items():
                tag = numeric_to_tag_dict.get(cid, cid)
                if len(pool) > 0:
                    print(f"[GC sampling pool] condition={cid} ({tag}): "
                          f"n={len(pool)}, min={pool.min()}, max={pool.max()}")
                else:
                    print(f"[GC sampling pool] condition={cid} ({tag}): n=0")
    else:
        gc_bin_pool_by_condition = None
        print("[Train] use_gc_condition=False, GC pool disabled.")

    train_dl, train_sampler = get_dataloader(
        dataset=train_data,
        batch_size=local_batch_size,
        num_workers=num_workers,
        distributed=distributed,
        pin_memory=pin_memory,
        random_seed=random_seed,
    )

    val_dl, _ = get_dataloader(
        dataset=val_data,
        batch_size=local_batch_size,
        num_workers=num_workers,
        distributed=distributed,
        pin_memory=pin_memory,
        random_seed=random_seed,
    )

    if rank_0 and use_wandb:
        init_wandb()

    global_step = 0
    model.train()

    best_val_loss = float("inf")
    val_patience_counter = 0
    checkpoint_files = []

    best_eval_score = float("-inf")
    eval_patience_counter = 0
    best_eval_file = None

    for epoch in tqdm(range(num_epochs), disable=not rank_0):
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        last_loss = None
        for batch in train_dl:
            if len(batch) == 3:
                x, y, gc_bin = batch
            else:
                raise ValueError("当前 train dataloader 应该返回 (x, y, gc_bin)。")

            loss = train_step(
                x=x, y=y, gc_bin=gc_bin,
                model=model, optimizer=optimizer, device=device,
                precision=precision, use_gc_condition=use_gc_condition,
            )
            last_loss = loss
            global_step += 1

            if rank_0 and global_step % log_step == 0 and use_wandb:
                wandb.log({"loss": loss, "epoch": epoch}, step=global_step)

        # Validation
        model.eval()
        val_losses = []
        for batch in val_dl:
            if len(batch) == 3:
                x, y, gc_bin = batch
            else:
                raise ValueError("当前 val dataloader 应该返回 (x, y, gc_bin)。")

            val_loss = val_step(
                x=x, y=y, gc_bin=gc_bin, model=model, device=device,
                precision=precision, use_gc_condition=use_gc_condition,
            )
            val_losses.append(val_loss)

        model.train()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")

        if rank_0 and use_wandb:
            wandb.log({
                "loss": last_loss if last_loss is not None else 0.0,
                "val_loss": avg_val_loss,
                "epoch": epoch,
            }, step=global_step)

        # ===== 额外评估 =====
        eval_metrics = {}
        eval_score = float("-inf")

        if rank_0 and ((epoch + 1) % eval_every_n_epochs == 0):
            eval_model = model.module if distributed else model

            # 💡 核心修改点：把完整的全局配置 global_cfg_dict 传入，替代原有的空字典
            eval_metrics = validate_diffusion_model(
                model=eval_model,
                data=data,
                device=torch.device(device),
                config=global_cfg_dict if global_cfg_dict is not None else {},
                predictor_eval=predictor_eval,
                num_samples_per_condition=eval_num_samples_per_condition,
                sample_bs=sample_batch_size,
                cond_weight=cond_weight, 
                sequence_length=sequence_length,  # 这里会作为备用默认值传递
                max_real_per_condition=500,
                alphabet="ACGT",
            )

            if eval_metrics:
                eval_score = compute_eval_score(eval_metrics)

                # 1. 基础生物学与通用指标（保持原样）
                base_keys = [
                    "3mer_R", "4mer_R", "5mer_R", "avg_corr", 
                    "gc_real", "gc_fake", "edit_dist", 
                    "pred_mean", "pred_std", "condition_acc", "condition_mae"
                ]
                msg_items = []
                for k in base_keys:
                    if k in eval_metrics and isinstance(eval_metrics[k], (int, float)):
                        msg_items.append(f"{k}: {eval_metrics[k]:.4f}")
                    else:
                        msg_items.append(f"{k}: N/A")

                # 2. 🚨 强制对齐打印三大核心命中率（就算是 0 也要现身！）
                hit_rate_keys = ["low_hit_rate", "mid_hit_rate", "high_hit_rate"]
                for k in hit_rate_keys:
                    val = eval_metrics.get(k, 0.0)  # 如果字典里没有，强行作为 0.0 处理
                    msg_items.append(f"{k}: {float(val):.4f}")

                # 打印第一行综合指标
                print("   [Eval] " + " | ".join(msg_items))

                # 3. 🚨 强制对齐打印三个分箱的预测均值（绝不漏掉 target_bin_0）
                per_bin_items = []
                for b_idx in [1, 2, 3]:
                    bin_key = f"target_bin_{b_idx}_pred_mean"
                    if bin_key in eval_metrics and isinstance(eval_metrics[bin_key], (int, float)):
                        per_bin_items.append(f"{bin_key}: {eval_metrics[bin_key]:.4f}")
                    else:
                        per_bin_items.append(f"{bin_key}: Empty(0)") # 如果这一轮挂零，明确提示

                print("   [PredMean by target bin] " + " | ".join(per_bin_items))

                print(f"   [EvalScore] composite_score: {eval_score:.4f}")

                if use_wandb:
                    wandb.log({
                        **{f"eval/{k}": v for k, v in eval_metrics.items()},
                        "eval/composite_score": eval_score,
                        "epoch": epoch,
                    }, step=global_step)

        # Save Best Model by val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            val_patience_counter = 0
            best_model_state, best_optimizer_state = get_current_state_dicts(model, optimizer, distributed)

            if rank_0:
                checkpoint_dict = {
                    "model": best_model_state, "optimizer": best_optimizer_state,
                    "epoch": epoch, "global_step": global_step, "val_loss": best_val_loss,
                    "eval_score": eval_score, "eval_metrics": eval_metrics,
                }
                checkpoint_file = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}_step{global_step}_valloss_{best_val_loss:.6f}.pt")
                torch.save(checkpoint_dict, checkpoint_file)
                checkpoint_files.append(checkpoint_file)

                if len(checkpoint_files) > 2:
                    old_file = checkpoint_files.pop(0)
                    if os.path.exists(old_file):
                        os.remove(old_file)
        else:
            val_patience_counter += 1

        # Save Best Model by composite score
        if eval_metrics and eval_score > best_eval_score:
            best_eval_score = eval_score
            eval_patience_counter = 0

            if rank_0:
                eval_model_state, eval_optimizer_state = get_current_state_dicts(model, optimizer, distributed)
                checkpoint_dict = {
                    "model": eval_model_state, "optimizer": eval_optimizer_state,
                    "epoch": epoch, "global_step": global_step, "val_loss": avg_val_loss,
                    "eval_score": best_eval_score, "eval_metrics": eval_metrics,
                }
                new_eval_file = os.path.join(CHECKPOINT_DIR, f"best_eval_epoch{epoch}_step{global_step}_score_{best_eval_score:.6f}.pt")
                torch.save(checkpoint_dict, new_eval_file)

                if best_eval_file is not None and os.path.exists(best_eval_file):
                    os.remove(best_eval_file)
                best_eval_file = new_eval_file
        elif eval_metrics:
            eval_patience_counter += 1

        # Early Stopping
        if epoch >= min_epochs and val_patience_counter >= patience and eval_patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} because both val_loss and eval_score have not improved.")
            print(f"Best val loss: {best_val_loss:.6f}")
            if best_eval_score > float("-inf"):
                print(f"Best composite eval score: {best_eval_score:.6f}")
            break

        # ===== 采样并保存中间生成的 DNA 序列 =====
        if rank_0 and (epoch + 1) % sample_epoch == 0:
            for cond_id in condition_id_list:
                create_sample(
                    model,
                    condition_id=cond_id,
                    sample_bs=sample_batch_size,
                    conditional_numeric_to_tag=numeric_to_tag_dict,
                    number_of_samples=number_of_samples,
                    cond_weight_to_metric=cond_weight,
                    save_timesteps=False,
                    save_dataframe=True,
                    generate_attention_maps=False,
                    sequence_length=sequence_length,  # 💡 此处已通过 main() 注入更新为正确的 165
                    gc_bin_pool_by_condition=gc_bin_pool_by_condition,
                    output_dir=SAMPLE_OUTPUT_DIR,
                )


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    allowed_train_keys = {
        "distributed", "precision", "num_workers", "pin_memory",
        "batch_size", "sample_batch_size", "log_step", "num_epochs",
        "min_epochs", "patience", "sample_epoch", "number_of_samples",
        "use_wandb", "eval_every_n_epochs", "eval_num_samples_per_condition",
        "sequence_length", "cond_weight", "random_seed",
    }

    train_setup = {
        k: v for k, v in cfg_dict["training"].items()
        if k in allowed_train_keys
    }

    train_setup.setdefault("random_seed", cfg_dict.get("data", {}).get("random_seed", 42))

    # 💡 核心修改点 1：显式从 cfg.data 中拉取正确的 sequence_length 注入 train_setup
    # 彻底杜绝 80 形参默认值在后续采样模块（例如 create_sample）引起的二级污染
    if "data" in cfg_dict and "sequence_length" in cfg_dict["data"]:
        train_setup["sequence_length"] = int(cfg_dict["data"]["sequence_length"])
        print(f"[Main] 从数据配置中成功读取并注入 sequence_length = {train_setup['sequence_length']}")

    use_gc = cfg.model.use_gc_condition
    data = hydra.utils.instantiate(cfg.data, use_gc_condition=use_gc)
    train_data = data[0]
    
    model = hydra.utils.instantiate(
        cfg.model,
        num_gc_bins=train_data.num_gc_bins if use_gc else None,
    )

    if hasattr(train_data, "num_gc_bins"):
        print("[Check] dataset num_gc_bins =", train_data.num_gc_bins)

    if "num_gc_bins" in cfg_dict.get("model", {}):
        print("[Check] model num_gc_bins =", cfg_dict["model"]["num_gc_bins"])
        if hasattr(train_data, "num_gc_bins"):
            if int(cfg_dict["model"]["num_gc_bins"]) != int(train_data.num_gc_bins):
                print(f"[WARN] cfg.model.num_gc_bins 和 dataset.num_gc_bins 不一致！")

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    diffusion = hydra.utils.instantiate(cfg.diffusion, model=model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor_cfg = dict(cfg_dict)
    predictor_cfg.update(cfg_dict.get("data", {}))
    predictor_cfg.update(cfg_dict.get("training", {}))

    predictor_eval = build_predictor_eval(predictor_cfg, device)

    # 💡 核心修改点 2：把融合后的 predictor_cfg 字典作为 global_cfg_dict 传递给 train
    # 确保里面的 predictor_eval_type 有明确值（例如 ecoli_lstm），激活内置长度双重防御
    train(
        **train_setup,
        model=diffusion,
        optimizer=optimizer,
        data=data,
        predictor_eval=predictor_eval,
        use_gc_condition=use_gc,  
        global_cfg_dict=predictor_cfg,  
    )


if __name__ == "__main__":
    main()