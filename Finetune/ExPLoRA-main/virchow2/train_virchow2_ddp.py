"""
Fine-tune Virchow2 ViT-H/14 with ExPLoRA, DINO/iBOT, and LoRA.
Supports torchrun on one or more GPUs, multiple data roots, BF16 AMP, gradient checkpointing, and epoch-level resume with EMA teacher state.
"""

import argparse
import os
import time
from datetime import datetime
from contextlib import nullcontext
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
from common.cli_config import load_config_file, merge_config_and_cli, select_section
from virchow2.modeling import (
    analyze_safetensors_compat,
    build_dino_ibot_head,
    build_backbone_from_virchow2_config,
    build_virchow2_backbone,
    compute_ema_momentum,
    ema_update,
    infer_ffn_layer_from_safetensors,
    infer_mlp_ratio_from_safetensors,
    inject_lora_and_freeze,
    load_virchow2_weights,
    maybe_load_virchow2_config,
)


def parse_args():
    p = argparse.ArgumentParser("Virchow2 ExPLoRA DDP finetune", add_help=True)
    # Accept a configuration file to support running without a long command line.
    p.add_argument(
        "--config",
        type=str,
        default="",
        help='JSON or YAML configuration. Command-line arguments take precedence.',
    )

    # Validate data and output paths after merging configuration and arguments.
    p.add_argument("--data_roots", type=str, nargs="+", default=None, help='One or more patch dataset roots')
    p.add_argument("--output_dir", type=str, default=None, help='Directory for checkpoints and logs')

    # Tar shards reduce per-file overhead, especially on WSL2 mounts.
    p.add_argument("--tar_index", type=str, default=None, help='Indexed tar dataset. Takes precedence over image directories.')
    p.add_argument(
        "--tar_shards_root",
        type=str,
        default=None,
        help='Base directory for relative shard paths. Defaults to the current directory.',
    )
    p.add_argument("--tar_max_samples", type=int, default=None, help='Limit the tar dataset size for debugging')
    p.add_argument(
        "--tar_group_by_shard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Group tar samples by shard by default. Disable with --no-tar-group-by-shard.',
    )
    p.add_argument("--tar_shard_cache_size", type=int, default=None, help='Number of open shards cached per DataLoader worker. Defaults to 32.')

    # Virchow2 weights and architecture configuration
    p.add_argument(
        "--safetensors_path",
        type=str,
        default=None,
        help='Virchow2 safetensors weights. Defaults to model.safetensors.',
    )
    p.add_argument(
        "--config_json",
        type=str,
        default=None,
        help='Virchow2 architecture config.json',
    )
    p.add_argument(
        "--check_weights_only",
        action="store_true",
        default=False,
        help='Check missing, unexpected, and mismatched checkpoint keys, then exit.',
    )
    p.add_argument(
        "--check_weights_topk",
        type=int,
        default=30,
        help='Example count per mismatch type in --check_weights_only mode',
    )

    # Training hyperparameters
    # Explicit arguments override the configuration.
    # Leave parser defaults as None so omitted arguments preserve configured values.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size_per_gpu", type=int, default=None)
    p.add_argument("--accumulation_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Keep DataLoader workers running between epochs',
    )
    p.add_argument("--prefetch_factor", type=int, default=None, help='DataLoader prefetch factor. Used when num_workers is positive.')
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--log_every",
        type=int,
        default=None,
        help='Number of optimizer steps between metric logs. Defaults to 50.',
    )
    p.add_argument(
        "--heartbeat_every",
        type=float,
        default=None,
        help=(
            'Seconds between training progress messages. '
            'Defaults to 30. Set 0 to disable.'
        ),
    )

    # AMP / precision
    p.add_argument("--amp_dtype", type=str, default=None, choices=["bf16", "fp16", "none"])
    # Enable gradient checkpointing by default and allow configuration overrides.
    p.add_argument(
        "--grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable backbone gradient checkpointing. Disable with --no-grad-ckpt.',
    )
    # Compatibility alias using underscores
    p.add_argument(
        "--no_grad_ckpt",
        dest="grad_ckpt",
        action="store_false",
        help='Compatibility alias for --no-grad-ckpt',
    )

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
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Resume training from a checkpoint. Disabled by default.',
    )
    p.add_argument("--resume_path", type=str, default="", help='Explicit checkpoint path for resume')

    # Read --config and merge explicit command-line overrides.
    pre, _ = p.parse_known_args()
    cfg_all = load_config_file(pre.config) if getattr(pre, "config", "") else {}
    cfg = select_section(cfg_all, "train_virchow2_ddp")

    cli = p.parse_args()
    merged = merge_config_and_cli(cfg, cli)

    # Apply defaults to options missing from both inputs.
    defaults = {
        # Virchow2 weights and architecture configuration
        "safetensors_path": "model.safetensors",
        "config_json": "virchow2/configs/virchow2_config.json",
        # Training hyperparameters
        "epochs": 50,
        "batch_size_per_gpu": 16,
        "accumulation_steps": 32,
        "lr": 2e-4,
        "num_workers": 8,
        "prefetch_factor": 4,
        "seed": 42,
        "log_every": 50,
        # Emit progress even when accumulation makes optimizer steps infrequent.
        "heartbeat_every": 30.0,
        # AMP / precision
        "amp_dtype": "bf16",
        # LoRA settings and trainable backbone layers
        "lora_rank": 16,
        "train_last_n_blocks": 2,
        # DINO/iBOT head
        "head_out_dim": 65536,
        "head_hidden_dim": 2048,
        "head_bottleneck_dim": 256,
        "head_nlayers": 3,
        # Loss / regularization
        "koleo_weight": 0.1,
        "ibot_weight": 1.0,
        "dino_weight": 1.0,
        "centering": "sinkhorn_knopp",
        "teacher_temp": 0.04,
        # iBOT masking
        "mask_ratio_min": 0.1,
        "mask_ratio_max": 0.5,
        "mask_probability": 0.5,
        # Teacher EMA
        "ema_base": 0.994,
        # tar dataset perf
        "tar_group_by_shard": True,
        "tar_shard_cache_size": 32,
    }
    for k, v in defaults.items():
        if merged.get(k, None) is None:
            merged[k] = v

    # Default to gradient checkpointing and a fresh run.
    if merged.get("grad_ckpt", None) is None:
        merged["grad_ckpt"] = True
    if merged.get("resume", None) is None:
        merged["resume"] = False
    # DataLoader: if user didn't specify, derive from num_workers to avoid torch ValueError
    if merged.get("persistent_workers", None) is None:
        merged["persistent_workers"] = bool(int(merged.get("num_workers", 0) or 0) > 0)

    # Validate required arguments after merging configuration.
    tar_index = merged.get("tar_index")
    tar_index = tar_index.strip() if isinstance(tar_index, str) else ""
    if not tar_index and not merged.get("data_roots"):
        p.error('Provide data_roots or tar_index. tar_index takes precedence.')
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
    # Use a timestamp that is valid in Windows filenames.
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_log_file(path: Path):
    global _LOG_FH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffer the training log for live progress.
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
            # Use the built-in print defaults for separator and terminator.
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
        # samples_list: [(aug_dict, meta), ...]
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


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Open log files on rank zero and allow console output on other ranks.
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

    # Infer the FFN from checkpoint keys before loading the weights.
    inferred_ffn = infer_ffn_layer_from_safetensors(args.safetensors_path, default="swiglufused")
    print0(rank, f"[weights] inferred ffn_layer={inferred_ffn!r} from safetensors keys")
    inferred_mlp_ratio = infer_mlp_ratio_from_safetensors(args.safetensors_path) if inferred_ffn == "mlp" else None
    if inferred_mlp_ratio is not None:
        print0(rank, f"[weights] inferred mlp_ratio={inferred_mlp_ratio:.6g} from safetensors fc2.weight")

    # --- Weights compatibility check (early exit) ---
    if getattr(args, "check_weights_only", False):
        vir_cfg = maybe_load_virchow2_config(getattr(args, "config_json", None))
        if vir_cfg is not None:
            backbone = build_backbone_from_virchow2_config(
                vir_cfg,
                block_chunks=0,
                ffn_layer=inferred_ffn,
                pos_embed_with_register_tokens=True,
                    mlp_ratio=inferred_mlp_ratio,
            )
        else:
            backbone = build_virchow2_backbone(
                img_size=224,
                patch_size=14,
                num_register_tokens=4,
                init_values=1e-5,
                block_chunks=0,
                pos_embed_with_register_tokens=False,
            )
        report = analyze_safetensors_compat(
            backbone,
            args.safetensors_path,
            topk=getattr(args, "check_weights_topk", 30),
        )
        if rank == 0:
            s = report["summary"]
            print("=== Virchow2 safetensors compatibility ===")
            print(f"safetensors_path: {args.safetensors_path}")
            print(f"config_json:      {getattr(args, 'config_json', '')}")
            print(
                "summary: "
                f"model_keys={s['num_model_keys']}  ckpt_keys={s['num_ckpt_keys']}  "
                f"matched={s['matched']}  missing={s['missing']}  "
                f"unexpected={s['unexpected']}  shape_mismatch={s['shape_mismatch']}"
            )
            topk = int(s.get("topk", 30))
            # shape mismatch
            sm = report["shape_mismatch"][:topk]
            print(f"shape_mismatch (showing {len(sm)}/{len(report['shape_mismatch'])}):")
            for (k, ckpt_shape, model_shape, ckpt_dtype, model_dtype) in sm:
                print(f"  - {k}: ckpt={ckpt_shape} {ckpt_dtype}  vs  model={model_shape} {model_dtype}")
            # missing/unexpected
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

    # Dataset / Aug
    aug = DataAugmentationDEXPloRA(local_crops_number=4, local_crops_size=98, global_crops_size=224)
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

    # iBOT masking config (global crops @224, patch_size=14 => 16x16 => 256 tokens)
    patch_size = 14
    global_size = 224
    n_tokens = (global_size // patch_size) * (global_size // patch_size)
    mask_gen = MaskingGenerator(input_size=(global_size // patch_size, global_size // patch_size))
    collate_fn = build_collate_fn(
        mask_ratio_tuple=(args.mask_ratio_min, args.mask_ratio_max),
        mask_probability=args.mask_probability,
        dtype=torch.float32,  # Collate inputs in float32 and apply autocast later.
        n_tokens=n_tokens,
        mask_generator=mask_gen,
    )

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

    # Model
    vir_cfg = maybe_load_virchow2_config(getattr(args, "config_json", None))
    if vir_cfg is not None:
        # Match Virchow2 positional embeddings for class, register, and patch tokens.
        backbone = build_backbone_from_virchow2_config(
            vir_cfg,
            block_chunks=0,
            ffn_layer=inferred_ffn,
            pos_embed_with_register_tokens=True,
            mlp_ratio=inferred_mlp_ratio,
        )
    else:
        # Fall back to the default architecture.
        backbone = build_virchow2_backbone(
            img_size=224,
            patch_size=14,
            num_register_tokens=4,
            init_values=1e-5,
            block_chunks=0,
            pos_embed_with_register_tokens=False,
        )
    missing, unexpected = load_virchow2_weights(backbone, args.safetensors_path, strict=False)
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
        in_dim=1280,
        out_dim=args.head_out_dim,
        bottleneck_dim=args.head_bottleneck_dim,
        hidden_dim=args.head_hidden_dim,
        nlayers=args.head_nlayers,
    )
    if vir_cfg is not None:
        teacher_backbone = build_backbone_from_virchow2_config(
            vir_cfg,
            block_chunks=0,
            # Match the student architecture so strict loading and EMA can succeed.
            ffn_layer=inferred_ffn,
            pos_embed_with_register_tokens=True,
            mlp_ratio=inferred_mlp_ratio,
        )
    else:
        teacher_backbone = build_virchow2_backbone(
            img_size=224,
            patch_size=14,
            num_register_tokens=4,
            init_values=1e-5,
            block_chunks=0,
            pos_embed_with_register_tokens=False,
        )
    # Use matching teacher and student architectures, including LoRA branches.
    inject_lora_and_freeze(
        teacher_backbone,
        lora_rank=args.lora_rank,
        train_last_n_blocks=0,
        include_attn_key=True,
    )
    teacher_backbone.load_state_dict(backbone.state_dict(), strict=True)
    teacher_head = build_dino_ibot_head(
        in_dim=1280,
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

    # Wrap the student in DDP.
    if world_size > 1:
        backbone = DDP(backbone, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        student_head = DDP(student_head, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    # Loss objects
    dino_loss = DINOLoss(args.head_out_dim)
    ibot_loss = iBOTPatchLoss(args.head_out_dim)
    koleo_loss = KoLeoLoss()

    # Use AdamW with zero weight decay.
    params = []
    for m in [backbone, student_head]:
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # Apply cosine scheduling at epoch granularity.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=0.0)

    # Save scaler state in the checkpoint, including in BF16 mode.
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

    steps_per_epoch = max(1, math.ceil(len(loader) / float(args.accumulation_steps)))
    max_steps = steps_per_epoch * max(args.epochs, 1)

    print0(rank, f"Start training for {args.epochs} epochs (start_epoch={start_epoch})")
    print0(rank, f"[run] steps_per_epoch={steps_per_epoch}, max_steps={max_steps}, log_every={args.log_every}")
    print0(rank, f"[run] heartbeat_every={float(getattr(args, 'heartbeat_every', 0.0) or 0.0):g}s")
    try:
        # Collect training statistics per optimizer step.
        last_step_end_t = time.time()
        avg_step_time = None  # EMA
        last_heartbeat_t = time.time()
        hb_every = float(getattr(args, "heartbeat_every", 0.0) or 0.0)
        # Log the current stage while loading data or running forward and backward passes.
        progress = {
            "epoch": int(start_epoch),
            "it": 0,
            "len_loader": int(len(loader)),
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
        # Reset peak memory tracking before each logging window.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            (backbone.module if isinstance(backbone, DDP) else backbone).train(True)
            (student_head.module if isinstance(student_head, DDP) else student_head).train(True)
            teacher_backbone.eval()
            teacher_head.eval()

            optimizer.zero_grad(set_to_none=True)
            t0 = time.time()
            loader_iter = iter(loader)
            n_loader = int(len(loader))
            for it in range(n_loader):
                progress["epoch"] = int(epoch)
                progress["it"] = int(it + 1)
                progress["len_loader"] = int(n_loader)
                progress["global_step"] = int(global_step)
                progress["max_steps"] = int(max_steps)
                progress["lr"] = float(optimizer.param_groups[0]["lr"])
                progress["stage"] = "waiting_data"
                batch = next(loader_iter)
                # batch keys:
                #   collated_global_crops: (2B,3,224,224)
                #   collated_local_crops:  (4B,3,98,98)
                #   collated_masks:        (2B,256) bool
                #   mask_indices_list:     (num_masked,) indices in flatten(2B*256)
                global_crops = batch["collated_global_crops"].to(device, non_blocking=True)
                local_crops = batch["collated_local_crops"].to(device, non_blocking=True)
                masks = batch["collated_masks"].to(device, non_blocking=True)
                mask_indices_list = batch["mask_indices_list"].to(device, non_blocking=True)
                masks_weight = batch["masks_weight"].to(device, non_blocking=True)
                n_masked_patches = int(batch["n_masked_patches"].item())

                # Teacher forward (global only)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                        progress["stage"] = "teacher_forward"
                        t_out = teacher_backbone(global_crops, is_training=True)
                        t_cls = t_out["x_norm_clstoken"]  # (2B,D)
                        t_patch = t_out["x_norm_patchtokens"]  # (2B,N,D)
                        t_cls_logits = teacher_head(t_cls)  # (2B,K)

                    if args.centering == "centering":
                        t_targets = dino_loss.softmax_center_teacher(t_cls_logits, teacher_temp=args.teacher_temp)
                        dino_loss.update_center(t_cls_logits)
                    else:
                        t_targets = dino_loss.sinkhorn_knopp_teacher(t_cls_logits, teacher_temp=args.teacher_temp)

                    # iBOT teacher masked patch targets
                    t_patch_flat = t_patch.flatten(0, 1)  # (2B*N, D)
                    t_masked = torch.index_select(t_patch_flat, dim=0, index=mask_indices_list)  # (num_masked,D)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                        t_masked_logits = teacher_head(t_masked)  # (num_masked,K)
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

                # Student forward (global+local)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    progress["stage"] = "student_forward"
                    s_out_g = (
                        backbone(global_crops, masks=masks, is_training=True)
                        if not isinstance(backbone, DDP)
                        else backbone(global_crops, masks=masks, is_training=True)
                    )
                    s_out_l = (
                        backbone(local_crops, masks=None, is_training=True)
                        if not isinstance(backbone, DDP)
                        else backbone(local_crops, masks=None, is_training=True)
                    )

                    s_cls_g = s_out_g["x_norm_clstoken"]  # (2B,D)
                    s_cls_l = s_out_l["x_norm_clstoken"]  # (4B,D)
                    s_patch_g = s_out_g["x_norm_patchtokens"]  # (2B,N,D)

                    s_cls_g_logits = (
                        student_head(s_cls_g) if not isinstance(student_head, DDP) else student_head(s_cls_g)
                    )  # (2B,K)
                    s_cls_l_logits = (
                        student_head(s_cls_l) if not isinstance(student_head, DDP) else student_head(s_cls_l)
                    )  # (4B,K)

                    s_patch_flat = s_patch_g.flatten(0, 1)  # (2B*N, D)
                    s_masked = torch.index_select(s_patch_flat, dim=0, index=mask_indices_list)  # (num_masked,D)
                    s_masked_logits = (
                        student_head(s_masked) if not isinstance(student_head, DDP) else student_head(s_masked)
                    )  # (num_masked,K)

                # DINO loss
                B = global_crops.shape[0] // 2
                t1, t2 = t_targets.chunk(2, dim=0)  # each (B,K)

                # local crops: (4B,K) -> 4 chunks
                n_local = 4
                s_local_list = list(s_cls_l_logits.chunk(n_local, dim=0))
                n_global_crops_loss_terms = 2  # (2-1)*2
                n_local_crops_loss_terms = max(n_local * 2, 1)
                denom = float(n_global_crops_loss_terms + n_local_crops_loss_terms)

                dino_local = (
                    dino_loss(student_output_list=s_local_list, teacher_out_softmaxed_centered_list=[t1, t2]) / denom
                )

                # global crops: align s_g1 -> t2, s_g2 -> t1
                s_g1, s_g2 = s_cls_g_logits.chunk(2, dim=0)
                teacher_swapped = torch.cat([t2, t1], dim=0)
                student_cat = torch.cat([s_g1, s_g2], dim=0)
                dino_global = dino_loss(
                    student_output_list=[student_cat], teacher_out_softmaxed_centered_list=[teacher_swapped]
                )
                dino_global = dino_global * 2.0 / denom

                # Compute KoLeo on backbone class-token embeddings.
                koleo = torch.tensor(0.0, device=device)
                if args.koleo_weight > 0:
                    s_cls_g_emb1, s_cls_g_emb2 = s_cls_g.chunk(2, dim=0)
                    koleo = args.koleo_weight * (koleo_loss(s_cls_g_emb1) + koleo_loss(s_cls_g_emb2))

                # iBOT masked patch loss
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

                # step
                do_step = ((it + 1) % args.accumulation_steps) == 0
                if do_step:
                    progress["stage"] = "optimizer_step"
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA update (teacher backbone + head)
                    m = compute_ema_momentum(args.ema_base, global_step, max_steps=max_steps)
                    sb = backbone.module if isinstance(backbone, DDP) else backbone
                    sh = student_head.module if isinstance(student_head, DDP) else student_head
                    ema_update(teacher_backbone, sb, m)
                    ema_update(teacher_head, sh, m)
                    global_step += 1
                    progress["global_step"] = int(global_step)
                    progress["stage"] = "running"

                    # Log metrics per optimizer step.
                    if args.log_every > 0 and (global_step % int(args.log_every) == 0):
                        # Average loss components across DDP ranks.
                        metrics = torch.stack(
                            [
                                loss_total.detach().float(),
                                dino_local.detach().float(),
                                dino_global.detach().float(),
                                ibot.detach().float(),
                                koleo.detach().float(),
                            ]
                        )  # (5,)
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                            metrics = metrics / float(world_size)

                        # Report the maximum peak memory across ranks in GB.
                        if torch.cuda.is_available():
                            peak_mem_gb = torch.tensor(
                                [torch.cuda.max_memory_allocated(device) / (1024**3)],
                                device=device,
                                dtype=torch.float32,
                            )
                        else:
                            peak_mem_gb = torch.tensor([0.0], device=device, dtype=torch.float32)
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(peak_mem_gb, op=dist.ReduceOp.MAX)

                        # Use rank-zero wall time for throughput and ETA.
                        now_t = time.time()
                        step_dt = now_t - last_step_end_t
                        last_step_end_t = now_t
                        if avg_step_time is None:
                            avg_step_time = step_dt
                        else:
                            avg_step_time = 0.9 * avg_step_time + 0.1 * step_dt

                        # Count unique samples covered by each optimizer step.
                        samples_per_step = int(args.batch_size_per_gpu) * int(args.accumulation_steps) * int(world_size)
                        # Each sample generates two global and four local crops.
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
                        # Reset peak memory for the next logging window.
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats(device)

            scheduler.step()

            # Save one timestamped checkpoint per epoch on rank zero.
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
                # Write a plain-text pointer to the latest checkpoint.
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
    # Allow the script to run as a Python module.
    main()


