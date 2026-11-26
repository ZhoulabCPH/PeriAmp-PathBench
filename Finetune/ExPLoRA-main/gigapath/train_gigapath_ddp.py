"""
Fine-tune the GigaPath tile encoder with ExPLoRA, DINO/iBOT, and LoRA.
The ViT-Giant backbone uses width 1536, 40 blocks, 24 heads, 16-pixel patches, zero register tokens, and packed SwiGLU. A 224-pixel crop yields 196 patch tokens. Local crops are 96 pixels.
Supports torchrun on one or more GPUs, multiple data roots, BF16 AMP, gradient checkpointing, and epoch-level resume with EMA teacher state.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple
import threading

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss

from common.augmentations import DataAugmentationDEXPloRA
from common.dataset import MultiCenterPathologyDataset
from common.tar_dataset import TarShardedPathologyDataset
from common.samplers import ShardGroupedDistributedSampler
from common.wds_dataset import build_wds_dataloader
from common.cli_config import load_config_file, merge_config_and_cli, select_section

import json as _json


def _read_wds_meta(wds_root: str):
    """Read num_samples and other metadata from the meta.json written by preprocess_tar_to_wds.py."""
    meta_path = Path(wds_root) / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return _json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


from gigapath.modeling import (
    analyze_pth_compat,
    build_backbone_from_gigapath_config,
    build_gigapath_backbone,
    compute_ema_momentum,
    ema_update,
    infer_ffn_layer_from_pth,
    infer_ffn_layer_from_state_dict,
    infer_mlp_ratio_from_pth,
    inject_lora_and_freeze,
    load_gigapath_weights,
    maybe_load_gigapath_config,
    build_dino_ibot_head,
)


def parse_args():
    p = argparse.ArgumentParser("GigaPath ExPLoRA DDP finetune", add_help=True)
    p.add_argument(
        "--config",
        type=str,
        default="",
        help='JSON or YAML configuration. Command-line arguments take precedence.',
    )

    p.add_argument("--data_roots", type=str, nargs="+", default=None, help='One or more patch dataset roots')
    p.add_argument("--output_dir", type=str, default=None, help='Directory for checkpoints and logs')

    p.add_argument("--tar_index", type=str, default=None, help='Tar dataset index.jsonl')
    p.add_argument("--tar_shards_root", type=str, default=None, help='Root directory for tar shards')
    p.add_argument("--tar_max_samples", type=int, default=None, help='Limit the tar dataset size for debugging')
    p.add_argument(
        "--tar_group_by_shard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Group tar samples by shard',
    )
    p.add_argument("--tar_shard_cache_size", type=int, default=None, help='Number of open shards cached per worker')

    # WebDataset takes precedence over the indexed tar dataset.
    p.add_argument("--wds_root", type=str, default=None,
                   help='WebDataset shard directory created by preprocess_tar_to_wds.py')
    p.add_argument("--wds_shard_list", type=str, default=None,
                   help='Shard list. Defaults to wds_root/shards.txt.')
    p.add_argument("--wds_epoch_length", type=int, default=None,
                   help='WebDataset samples per epoch. Defaults to automatic detection.')
    p.add_argument("--wds_shuffle_buffer", type=int, default=None,
                   help='WebDataset sample shuffle buffer size')

    # GigaPath weights and architecture configuration
    p.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help='GigaPath tile encoder weights (.pth)',
    )
    p.add_argument(
        "--config_json",
        type=str,
        default=None,
        help='GigaPath architecture config.json',
    )
    p.add_argument("--check_weights_only", action="store_true", default=False)
    p.add_argument("--check_weights_topk", type=int, default=30)

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size_per_gpu", type=int, default=None)
    p.add_argument("--accumulation_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--heartbeat_every", type=float, default=None)

    # AMP / precision
    p.add_argument("--amp_dtype", type=str, default=None, choices=["bf16", "fp16", "none"])
    p.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")

    # LoRA settings and trainable backbone layers
    p.add_argument("--lora_rank", type=int, default=None)
    p.add_argument("--train_last_n_blocks", type=int, default=None)

    # DINO/iBOT head
    p.add_argument("--head_out_dim", type=int, default=None)
    p.add_argument("--head_hidden_dim", type=int, default=None)
    p.add_argument("--head_bottleneck_dim", type=int, default=None)
    p.add_argument("--head_nlayers", type=int, default=None)

    # Loss / regularization
    p.add_argument("--koleo_weight", type=float, default=None)
    p.add_argument("--ibot_weight", type=float, default=None)
    p.add_argument("--dino_weight", type=float, default=None)
    p.add_argument("--centering", type=str, default=None, choices=["centering", "sinkhorn_knopp"])
    p.add_argument("--teacher_temp", type=float, default=None)

    # iBOT masking
    p.add_argument("--mask_ratio_min", type=float, default=None)
    p.add_argument("--mask_ratio_max", type=float, default=None)
    p.add_argument("--mask_probability", type=float, default=None)

    # Teacher EMA
    p.add_argument("--ema_base", type=float, default=None)

    # Checkpoint
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--resume_path", type=str, default="")

    pre, _ = p.parse_known_args()
    cfg_all = load_config_file(pre.config) if getattr(pre, "config", "") else {}
    cfg = select_section(cfg_all, "train_gigapath_ddp")

    cli = p.parse_args()
    merged = merge_config_and_cli(cfg, cli)

    defaults = {
        "weights_path": "prov-gigapath-tile-encoder.pth",
        "config_json": "gigapath/configs/gigapath_config.json",
        "epochs": 50,
        "batch_size_per_gpu": 16,
        "accumulation_steps": 32,
        "lr": 2e-4,
        "num_workers": 8,
        "prefetch_factor": 4,
        "seed": 42,
        "log_every": 50,
        "heartbeat_every": 30.0,
        "amp_dtype": "bf16",
        "lora_rank": 16,
        "train_last_n_blocks": 2,
        "head_out_dim": 65536,
        "head_hidden_dim": 2048,
        "head_bottleneck_dim": 256,
        "head_nlayers": 3,
        "koleo_weight": 0.1,
        "ibot_weight": 1.0,
        "dino_weight": 1.0,
        "centering": "sinkhorn_knopp",
        "teacher_temp": 0.04,
        "mask_ratio_min": 0.1,
        "mask_ratio_max": 0.5,
        "mask_probability": 0.5,
        "ema_base": 0.994,
        "tar_group_by_shard": True,
        "tar_shard_cache_size": 32,
        "wds_shuffle_buffer": 5000,
    }
    for k, v in defaults.items():
        if merged.get(k, None) is None:
            merged[k] = v

    if merged.get("grad_ckpt", None) is None:
        merged["grad_ckpt"] = True
    if merged.get("resume", None) is None:
        merged["resume"] = False
    if merged.get("persistent_workers", None) is None:
        merged["persistent_workers"] = bool(int(merged.get("num_workers", 0) or 0) > 0)

    tar_index = merged.get("tar_index")
    tar_index = tar_index.strip() if isinstance(tar_index, str) else ""
    wds_root = merged.get("wds_root")
    wds_root = wds_root.strip() if isinstance(wds_root, str) else ""
    if not wds_root and not tar_index and not merged.get("data_roots"):
        p.error('Provide wds_root, tar_index, or data_roots. wds_root takes precedence.')
    if not merged.get("output_dir"):
        p.error('Provide --output_dir or set output_dir in the configuration.')

    return argparse.Namespace(**merged)


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> Tuple[int, int, int]:
    if not is_dist():
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


_LOG_FH = None


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_log_file(path: Path):
    global _LOG_FH
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(path, "a", encoding="utf-8", buffering=1)


def _close_log_file():
    global _LOG_FH
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        finally:
            _LOG_FH = None


def print0(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)
        if _LOG_FH is not None:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            msg = sep.join(str(a) for a in args) + end
            _LOG_FH.write(msg)


def seed_all(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast_dtype(amp_dtype: str):
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    return None


def build_collate_fn(mask_ratio_tuple, mask_probability, dtype, n_tokens, mask_generator):
    def _fn(samples_list):
        return collate_data_and_cast(
            samples_list,
            mask_ratio_tuple=mask_ratio_tuple,
            mask_probability=mask_probability,
            dtype=dtype,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
        )
    return _fn


def try_resume(
    args,
    rank: int,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    teacher_backbone: torch.nn.Module,
    teacher_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
):
    if not args.resume and not args.resume_path:
        return 0, 0

    def _pick_latest_ckpt(out_dir: Path) -> Path:
        candidates = []
        legacy = out_dir / "checkpoint.pth"
        if legacy.exists():
            candidates.append(legacy)
        ckpt_dir = out_dir / "checkpoints"
        if ckpt_dir.exists():
            candidates.extend([p for p in ckpt_dir.glob("*.pth") if p.is_file() and not p.name.endswith(".tmp")])
        if not candidates:
            return Path("")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    if args.resume_path.strip():
        path = Path(args.resume_path.strip())
    else:
        path = _pick_latest_ckpt(Path(args.output_dir))

    if (not path) or (not path.exists()):
        print0(rank, f"[resume] No checkpoint found in {args.output_dir}. Starting a new run.")
        return 0, 0

    ckpt = torch.load(str(path), map_location="cpu")
    backbone.load_state_dict(ckpt["model"]["backbone"], strict=True)
    head.load_state_dict(ckpt["model"]["head"], strict=True)
    teacher_backbone.load_state_dict(ckpt["teacher"]["backbone"], strict=True)
    teacher_head.load_state_dict(ckpt["teacher"]["head"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    try:
        scaler.load_state_dict(ckpt["scaler"])
    except Exception:
        pass
    try:
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception:
        pass
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    print0(rank, f"[resume] Resumed from {str(path)} at start_epoch={start_epoch}, global_step={global_step}")
    return start_epoch, global_step


# GigaPath architecture constants
GIGAPATH_EMBED_DIM = 1536
GIGAPATH_PATCH_SIZE = 16
GIGAPATH_GLOBAL_CROP_SIZE = 224
GIGAPATH_LOCAL_CROP_SIZE = 96  # 96 pixels gives six tokens per side.


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    run_stamp = _now_stamp()
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        _set_log_file(out_dir / "logs" / f"train_{run_stamp}.txt")
        print0(rank, f"[run] start_time={run_stamp}")
        print0(rank, f"[run] output_dir={str(out_dir.resolve())}")
        print0(rank, f"[run] args={vars(args)}")

    seed_all(args.seed + rank)

    eff_bs = args.batch_size_per_gpu * args.accumulation_steps * world_size
    print0(rank, f"DDP: world_size={world_size}, local_rank={local_rank}")
    print0(rank, f"Effective batch size = {eff_bs} (bs/gpu={args.batch_size_per_gpu}, accum={args.accumulation_steps})")

    # Infer the FFN type from the GigaPath weights.
    inferred_ffn = infer_ffn_layer_from_pth(args.weights_path, default="swiglupacked")
    print0(rank, f"[weights] inferred ffn_layer={inferred_ffn!r} from .pth keys")
    inferred_mlp_ratio = infer_mlp_ratio_from_pth(args.weights_path) if inferred_ffn == "mlp" else None
    if inferred_mlp_ratio is not None:
        print0(rank, f"[weights] inferred mlp_ratio={inferred_mlp_ratio:.6g}")

    # --- Weights compatibility check (early exit) ---
    if getattr(args, "check_weights_only", False):
        gp_cfg = maybe_load_gigapath_config(getattr(args, "config_json", None))
        if gp_cfg is not None:
            backbone = build_backbone_from_gigapath_config(
                gp_cfg,
                block_chunks=0,
                ffn_layer=inferred_ffn,
                pos_embed_with_register_tokens=False,
                mlp_ratio=inferred_mlp_ratio,
            )
        else:
            backbone = build_gigapath_backbone(ffn_layer=inferred_ffn)
        report = analyze_pth_compat(backbone, args.weights_path, topk=getattr(args, "check_weights_topk", 30))
        if rank == 0:
            s = report["summary"]
            print("=== GigaPath .pth compatibility ===")
            print(f"weights_path: {args.weights_path}")
            print(f"config_json:  {getattr(args, 'config_json', '')}")
            print(
                "summary: "
                f"model_keys={s['num_model_keys']}  ckpt_keys={s['num_ckpt_keys']}  "
                f"matched={s['matched']}  missing={s['missing']}  "
                f"unexpected={s['unexpected']}  shape_mismatch={s['shape_mismatch']}"
            )
            topk = int(s.get("topk", 30))
            sm = report["shape_mismatch"][:topk]
            print(f"shape_mismatch (showing {len(sm)}/{len(report['shape_mismatch'])}):")
            for (k, ckpt_shape, model_shape, ckpt_dtype, model_dtype) in sm:
                print(f"  - {k}: ckpt={ckpt_shape} {ckpt_dtype}  vs  model={model_shape} {model_dtype}")
            mk = report["missing_keys"][:topk]
            uk = report["unexpected_keys"][:topk]
            print(f"missing_keys (showing {len(mk)}/{len(report['missing_keys'])}):")
            for k in mk:
                print(f"  - {k}")
            print(f"unexpected_keys (showing {len(uk)}/{len(report['unexpected_keys'])}):")
            for k in uk:
                print(f"  - {k}")
        cleanup_distributed()
        _close_log_file()
        return

    # Use shared augmentations with 96-pixel local crops for the 16-pixel patch grid.
    aug = DataAugmentationDEXPloRA(
        local_crops_number=4,
        local_crops_size=GIGAPATH_LOCAL_CROP_SIZE,
        global_crops_size=GIGAPATH_GLOBAL_CROP_SIZE,
    )

    # iBOT masking: GigaPath patch_size=16 => 224/16=14 => 14x14 = 196 tokens
    patch_size = GIGAPATH_PATCH_SIZE
    global_size = GIGAPATH_GLOBAL_CROP_SIZE
    n_tokens = (global_size // patch_size) * (global_size // patch_size)
    mask_gen = MaskingGenerator(input_size=(global_size // patch_size, global_size // patch_size))
    collate_fn = build_collate_fn(
        mask_ratio_tuple=(args.mask_ratio_min, args.mask_ratio_max),
        mask_probability=args.mask_probability,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_gen,
    )

    use_wds = isinstance(getattr(args, "wds_root", None), str) and args.wds_root.strip()
    sampler = None

    if use_wds:
        print0(rank, f"[data] Streaming WebDataset from {args.wds_root}")

        wds_epoch_length = getattr(args, "wds_epoch_length", None)
        if wds_epoch_length is not None:
            wds_epoch_length = int(wds_epoch_length) if int(wds_epoch_length) > 0 else None

        if wds_epoch_length is None:
            wds_meta = _read_wds_meta(args.wds_root)
            if wds_meta is not None and "num_samples" in wds_meta:
                total_samples = int(wds_meta["num_samples"])
                per_rank = total_samples // max(world_size, 1)
                wds_epoch_length = max(1, (per_rank // args.batch_size_per_gpu) * args.batch_size_per_gpu)
                print0(rank, f"[wds] Inferred epoch_length from meta.json: "
                       f"num_samples={total_samples}, world_size={world_size}, "
                       f"batch_size={args.batch_size_per_gpu} → epoch_length={wds_epoch_length}")
            else:
                print0(rank, '[wds] Cannot infer epoch_length. meta.json or num_samples is missing.')

        loader = build_wds_dataloader(
            wds_root=args.wds_root,
            transform=aug,
            collate_fn=collate_fn,
            shard_list=getattr(args, "wds_shard_list", None) or "shards.txt",
            epoch_length=wds_epoch_length,
            shuffle_buffer=int(getattr(args, "wds_shuffle_buffer", 5000) or 5000),
            num_workers=args.num_workers,
            batch_size=args.batch_size_per_gpu,
            prefetch_factor=int(getattr(args, "prefetch_factor", 4) or 4),
            world_size=world_size,
            rank=rank,
            seed=args.seed,
            pin_memory=True,
            persistent_workers=bool(getattr(args, "persistent_workers", True)),
        )
    else:
        if isinstance(getattr(args, "tar_index", None), str) and args.tar_index.strip():
            dataset = TarShardedPathologyDataset(
                index_path=args.tar_index,
                shards_root=getattr(args, "tar_shards_root", None),
                transform=aug,
                max_samples=getattr(args, "tar_max_samples", None),
                shard_cache_size=int(getattr(args, "tar_shard_cache_size", 32) or 32),
            )
        else:
            dataset = MultiCenterPathologyDataset(args.data_roots, transform=aug)
        if world_size > 1:
            if isinstance(dataset, TarShardedPathologyDataset) and bool(getattr(args, "tar_group_by_shard", True)):
                sampler = ShardGroupedDistributedSampler(
                    records=dataset.records,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=int(getattr(args, "seed", 0) or 0),
                    drop_last=False,
                )
            else:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            sampler = None

        dl_kwargs = {}
        if int(getattr(args, "num_workers", 0) or 0) > 0:
            dl_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
            dl_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4) or 4)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size_per_gpu,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            **dl_kwargs,
        )

    # Model (GigaPath ViT-Giant)
    gp_cfg = maybe_load_gigapath_config(getattr(args, "config_json", None))
    if gp_cfg is not None:
        backbone = build_backbone_from_gigapath_config(
            gp_cfg,
            block_chunks=0,
            ffn_layer=inferred_ffn,
            pos_embed_with_register_tokens=False,
            mlp_ratio=inferred_mlp_ratio,
        )
    else:
        backbone = build_gigapath_backbone(ffn_layer=inferred_ffn)

    missing, unexpected = load_gigapath_weights(backbone, args.weights_path, strict=False)
    print0(rank, f"[weights] missing={len(missing)} unexpected={len(unexpected)}")

    if args.grad_ckpt:
        backbone.set_gradient_checkpointing(True)

    inject_lora_and_freeze(
        backbone,
        lora_rank=args.lora_rank,
        train_last_n_blocks=args.train_last_n_blocks,
        include_attn_key=True,
    )

    student_head = build_dino_ibot_head(
        in_dim=GIGAPATH_EMBED_DIM,
        out_dim=args.head_out_dim,
        bottleneck_dim=args.head_bottleneck_dim,
        hidden_dim=args.head_hidden_dim,
        nlayers=args.head_nlayers,
    )

    if gp_cfg is not None:
        teacher_backbone = build_backbone_from_gigapath_config(
            gp_cfg,
            block_chunks=0,
            ffn_layer=inferred_ffn,
            pos_embed_with_register_tokens=False,
            mlp_ratio=inferred_mlp_ratio,
        )
    else:
        teacher_backbone = build_gigapath_backbone(ffn_layer=inferred_ffn)

    inject_lora_and_freeze(
        teacher_backbone,
        lora_rank=args.lora_rank,
        train_last_n_blocks=0,
        include_attn_key=True,
    )
    teacher_backbone.load_state_dict(backbone.state_dict(), strict=True)
    teacher_head = build_dino_ibot_head(
        in_dim=GIGAPATH_EMBED_DIM,
        out_dim=args.head_out_dim,
        bottleneck_dim=args.head_bottleneck_dim,
        hidden_dim=args.head_hidden_dim,
        nlayers=args.head_nlayers,
    )
    teacher_head.load_state_dict(student_head.state_dict(), strict=True)

    for p in teacher_backbone.parameters():
        p.requires_grad_(False)
    for p in teacher_head.parameters():
        p.requires_grad_(False)

    backbone.to(device)
    student_head.to(device)
    teacher_backbone.to(device)
    teacher_head.to(device)

    if world_size > 1:
        backbone = DDP(backbone, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        student_head = DDP(student_head, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    # Loss
    dino_loss = DINOLoss(args.head_out_dim)
    ibot_loss = iBOTPatchLoss(args.head_out_dim)
    koleo_loss = KoLeoLoss()

    # Optimizer
    params = []
    for m in [backbone, student_head]:
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=0.0)
    amp_dtype = get_autocast_dtype(args.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_dtype == "fp16"))

    # Resume
    sb = backbone.module if isinstance(backbone, DDP) else backbone
    sh = student_head.module if isinstance(student_head, DDP) else student_head
    start_epoch, global_step = try_resume(
        args,
        rank=rank,
        backbone=sb,
        head=sh,
        teacher_backbone=teacher_backbone,
        teacher_head=teacher_head,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
    )

    import math

    if use_wds:
        if wds_epoch_length is not None and wds_epoch_length > 0:
            _loader_len = max(1, wds_epoch_length // args.batch_size_per_gpu)
        else:
            _loader_len = 0
    else:
        _loader_len = len(loader)

    steps_per_epoch = max(1, math.ceil(_loader_len / float(args.accumulation_steps))) if _loader_len > 0 else 0
    max_steps = steps_per_epoch * max(args.epochs, 1) if steps_per_epoch > 0 else 0

    print0(rank, f"Start training for {args.epochs} epochs (start_epoch={start_epoch})")
    print0(rank, f"[run] steps_per_epoch={steps_per_epoch}, max_steps={max_steps}, log_every={args.log_every}")
    if use_wds:
        print0(rank, f"[run] WebDataset epoch_length={wds_epoch_length}, "
               f"loader_len(batches/rank)={_loader_len}")
    print0(rank, f"[run] heartbeat_every={float(getattr(args, 'heartbeat_every', 0.0) or 0.0):g}s")
    try:
        last_step_end_t = time.time()
        avg_step_time = None
        hb_every = float(getattr(args, "heartbeat_every", 0.0) or 0.0)
        progress = {
            "epoch": int(start_epoch),
            "it": 0,
            "len_loader": int(_loader_len),
            "global_step": int(global_step),
            "max_steps": int(max_steps),
            "stage": "init",
            "lr": 0.0,
        }
        hb_stop = threading.Event()

        def _hb_loop():
            while not hb_stop.is_set():
                hb_stop.wait(timeout=hb_every)
                if hb_stop.is_set():
                    break
                try:
                    if torch.cuda.is_available():
                        mem_gb = float(torch.cuda.memory_allocated(device) / (1024**3))
                    else:
                        mem_gb = 0.0
                    print0(
                        rank,
                        "[hb]"
                        f" epoch={progress['epoch']}"
                        f" iter={progress['it']}/{progress['len_loader']}"
                        f" step={progress['global_step']}/{progress['max_steps']}"
                        f" lr={progress['lr']:.3e}"
                        f" stage={progress['stage']}"
                        f" mem={mem_gb:.2f}GB",
                    )
                except Exception:
                    pass

        hb_thread = None
        if rank == 0 and hb_every > 0:
            hb_thread = threading.Thread(target=_hb_loop, name="train_heartbeat", daemon=True)
            hb_thread.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(start_epoch, args.epochs):
            if not use_wds and sampler is not None:
                sampler.set_epoch(epoch)

            (backbone.module if isinstance(backbone, DDP) else backbone).train(True)
            (student_head.module if isinstance(student_head, DDP) else student_head).train(True)
            teacher_backbone.eval()
            teacher_head.eval()

            optimizer.zero_grad(set_to_none=True)
            t0 = time.time()
            loader_iter = iter(loader)
            n_loader = int(_loader_len) if _loader_len > 0 else 0
            it = 0
            while True:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    if use_wds and n_loader == 0:
                        n_loader = it
                    break
                progress["epoch"] = int(epoch)
                progress["it"] = int(it + 1)
                progress["len_loader"] = int(n_loader) if n_loader > 0 else it + 1
                progress["global_step"] = int(global_step)
                progress["max_steps"] = int(max_steps)
                progress["lr"] = float(optimizer.param_groups[0]["lr"])
                progress["stage"] = "waiting_data"
                if not use_wds and it >= n_loader:
                    break

                global_crops = batch["collated_global_crops"].to(device, non_blocking=True)
                local_crops = batch["collated_local_crops"].to(device, non_blocking=True)
                masks = batch["collated_masks"].to(device, non_blocking=True)
                mask_indices_list = batch["mask_indices_list"].to(device, non_blocking=True)
                masks_weight = batch["masks_weight"].to(device, non_blocking=True)
                n_masked_patches = int(batch["n_masked_patches"].item())

                # Teacher forward
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                        progress["stage"] = "teacher_forward"
                        t_out = teacher_backbone(global_crops, is_training=True)
                        t_cls = t_out["x_norm_clstoken"]
                        t_patch = t_out["x_norm_patchtokens"]
                        t_cls_logits = teacher_head(t_cls)

                    if args.centering == "centering":
                        t_targets = dino_loss.softmax_center_teacher(t_cls_logits, teacher_temp=args.teacher_temp)
                        dino_loss.update_center(t_cls_logits)
                    else:
                        t_targets = dino_loss.sinkhorn_knopp_teacher(t_cls_logits, teacher_temp=args.teacher_temp)

                    t_patch_flat = t_patch.flatten(0, 1)
                    t_masked = torch.index_select(t_patch_flat, dim=0, index=mask_indices_list)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                        t_masked_logits = teacher_head(t_masked)
                    if args.centering == "centering":
                        t_patch_targets = ibot_loss.softmax_center_teacher(
                            t_masked_logits.unsqueeze(0), teacher_temp=args.teacher_temp
                        )
                        t_patch_targets = t_patch_targets.squeeze(0)
                        ibot_loss.update_center(t_masked_logits.unsqueeze(0))
                    else:
                        t_patch_targets = ibot_loss.sinkhorn_knopp_teacher(
                            t_masked_logits,
                            teacher_temp=args.teacher_temp,
                            n_masked_patches_tensor=batch["n_masked_patches"].to(device),
                        )

                # Student forward
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    progress["stage"] = "student_forward"
                    s_out_g = backbone(global_crops, masks=masks, is_training=True)
                    s_out_l = backbone(local_crops, masks=None, is_training=True)

                    s_cls_g = s_out_g["x_norm_clstoken"]
                    s_cls_l = s_out_l["x_norm_clstoken"]
                    s_patch_g = s_out_g["x_norm_patchtokens"]

                    s_cls_g_logits = student_head(s_cls_g)
                    s_cls_l_logits = student_head(s_cls_l)

                    s_patch_flat = s_patch_g.flatten(0, 1)
                    s_masked = torch.index_select(s_patch_flat, dim=0, index=mask_indices_list)
                    s_masked_logits = student_head(s_masked)

                # DINO loss
                B = global_crops.shape[0] // 2
                t1, t2 = t_targets.chunk(2, dim=0)

                n_local = 4
                s_local_list = list(s_cls_l_logits.chunk(n_local, dim=0))
                n_global_crops_loss_terms = 2
                n_local_crops_loss_terms = max(n_local * 2, 1)
                denom = float(n_global_crops_loss_terms + n_local_crops_loss_terms)

                dino_local = (
                    dino_loss(student_output_list=s_local_list, teacher_out_softmaxed_centered_list=[t1, t2]) / denom
                )

                s_g1, s_g2 = s_cls_g_logits.chunk(2, dim=0)
                teacher_swapped = torch.cat([t2, t1], dim=0)
                student_cat = torch.cat([s_g1, s_g2], dim=0)
                dino_global = dino_loss(
                    student_output_list=[student_cat], teacher_out_softmaxed_centered_list=[teacher_swapped]
                )
                dino_global = dino_global * 2.0 / denom

                koleo = torch.tensor(0.0, device=device)
                if args.koleo_weight > 0:
                    s_cls_g_emb1, s_cls_g_emb2 = s_cls_g.chunk(2, dim=0)
                    koleo = args.koleo_weight * (koleo_loss(s_cls_g_emb1) + koleo_loss(s_cls_g_emb2))

                ibot = torch.tensor(0.0, device=device)
                if args.ibot_weight > 0:
                    ibot = ibot_loss.forward_masked(
                        student_patch_tokens_masked=s_masked_logits,
                        teacher_patch_tokens_masked=t_patch_targets,
                        student_masks_flat=masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    ibot = args.ibot_weight * ibot

                loss_total = args.dino_weight * (dino_local + dino_global) + koleo + ibot
                loss_for_backward = loss_total / float(args.accumulation_steps)

                if scaler.is_enabled():
                    progress["stage"] = "backward(fp16)"
                    scaler.scale(loss_for_backward).backward()
                else:
                    progress["stage"] = "backward"
                    loss_for_backward.backward()

                do_step = ((it + 1) % args.accumulation_steps) == 0
                if do_step:
                    progress["stage"] = "optimizer_step"
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    m = compute_ema_momentum(args.ema_base, global_step, max_steps=max_steps)
                    sb = backbone.module if isinstance(backbone, DDP) else backbone
                    sh = student_head.module if isinstance(student_head, DDP) else student_head
                    ema_update(teacher_backbone, sb, m)
                    ema_update(teacher_head, sh, m)
                    global_step += 1
                    progress["global_step"] = int(global_step)
                    progress["stage"] = "running"

                    if args.log_every > 0 and (global_step % int(args.log_every) == 0):
                        metrics = torch.stack([
                            loss_total.detach().float(),
                            dino_local.detach().float(),
                            dino_global.detach().float(),
                            ibot.detach().float(),
                            koleo.detach().float(),
                        ])
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                            metrics = metrics / float(world_size)

                        if torch.cuda.is_available():
                            peak_mem_gb = torch.tensor(
                                [torch.cuda.max_memory_allocated(device) / (1024**3)],
                                device=device, dtype=torch.float32,
                            )
                        else:
                            peak_mem_gb = torch.tensor([0.0], device=device, dtype=torch.float32)
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(peak_mem_gb, op=dist.ReduceOp.MAX)

                        now_t = time.time()
                        step_dt = now_t - last_step_end_t
                        last_step_end_t = now_t
                        if avg_step_time is None:
                            avg_step_time = step_dt
                        else:
                            avg_step_time = 0.9 * avg_step_time + 0.1 * step_dt

                        samples_per_step = int(args.batch_size_per_gpu) * int(args.accumulation_steps) * int(world_size)
                        crops_per_step = samples_per_step * 6
                        sps = samples_per_step / max(step_dt, 1e-9)
                        cps = crops_per_step / max(step_dt, 1e-9)

                        remain = max(int(max_steps) - int(global_step), 0)
                        eta_sec = float(avg_step_time) * float(remain) if avg_step_time is not None else 0.0
                        eta_h = int(eta_sec // 3600)
                        eta_m = int((eta_sec % 3600) // 60)
                        eta_s = int(eta_sec % 60)

                        print0(
                            rank,
                            "[train]"
                            f" time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            f" epoch={epoch}"
                            f" step={global_step}/{max_steps}"
                            f" lr={optimizer.param_groups[0]['lr']:.3e}"
                            f" loss={metrics[0].item():.4f}"
                            f" (dino_local={metrics[1].item():.4f}, dino_global={metrics[2].item():.4f}, ibot={metrics[3].item():.4f}, koleo={metrics[4].item():.4f})"
                            f" step_time={step_dt:.2f}s"
                            f" throughput={sps:.1f} samples/s ({cps:.1f} crops/s)"
                            f" peak_mem={peak_mem_gb.item():.2f}GB"
                            f" ETA={eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats(device)

                it += 1

            if use_wds and _loader_len == 0 and n_loader > 0:
                _loader_len = n_loader
                steps_per_epoch = max(1, math.ceil(_loader_len / float(args.accumulation_steps)))
                max_steps = steps_per_epoch * max(args.epochs, 1)
                progress["len_loader"] = int(_loader_len)
                progress["max_steps"] = int(max_steps)
                print0(rank, f"[wds] epoch {epoch}  Updated with actual iteration count: {n_loader}, "
                       f"steps_per_epoch={steps_per_epoch}, max_steps={max_steps}")

            scheduler.step()

            if rank == 0:
                sb = backbone.module if isinstance(backbone, DDP) else backbone
                sh = student_head.module if isinstance(student_head, DDP) else student_head
                model_state = {"backbone": sb.state_dict(), "head": sh.state_dict()}
                teacher_state = {"backbone": teacher_backbone.state_dict(), "head": teacher_head.state_dict()}
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model_state,
                    "teacher": teacher_state,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                }
                ckpt_dir = out_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                stamp = _now_stamp()
                final = ckpt_dir / f"checkpoint_{stamp}_epoch{epoch:04d}_gs{global_step}.pth"
                tmp = final.with_suffix(".pth.tmp")
                torch.save(ckpt, tmp)
                tmp.replace(final)
                (out_dir / "latest_checkpoint.txt").write_text(str(final), encoding="utf-8")
                print0(rank, f"[ckpt] saved: {str(final)}")

            if dist.is_initialized():
                dist.barrier()

            dt = time.time() - t0
            print0(rank, f"Epoch {epoch} done. time={dt:.1f}s, lr={optimizer.param_groups[0]['lr']:.3e}, global_step={global_step}")
    finally:
        try:
            hb_stop.set()
            if hb_thread is not None:
                hb_thread.join(timeout=2.0)
        except Exception:
            pass
        cleanup_distributed()
        if rank == 0:
            print0(rank, "[run] finished")
            _close_log_file()


if __name__ == "__main__":
    main()
