"""
Merge LoRA into the GigaPath backbone and export standalone weights.
Apply W = W0 + BA to the adapted linear layers and combine the Q, K, and V projections into QKV.
"""

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dinov2.utils.train_lora_util import deactivate_lora, lora_merge_all  # noqa: E402

from gigapath.modeling import (  # noqa: E402
    build_gigapath_backbone,
    inject_lora_and_freeze,
    infer_ffn_layer_from_state_dict,
    infer_mlp_ratio_from_state_dict,
)
from common.cli_config import (  # noqa: E402
    load_config_file,
    merge_config_and_cli,
    select_section,
)


def parse_args():
    p = argparse.ArgumentParser("Export merged GigaPath backbone", add_help=True)
    p.add_argument("--config",
                   type=str,
                   default=r"gigapath/configs/gigapath_config.json",
                   help='Configuration file path')

    p.add_argument("--checkpoint",
                   type=str,
                   default=r"weights/gigapath_checkpoint.pth",
                   help='Training checkpoint (.pth)')
    p.add_argument("--output",
                   type=str,
                   default=r".\outputs\gigapath_explora\Late Stage(teacher)/vit_giant_patch14_dinov2.pth",
                   help='Output path for the merged backbone weights')
    p.add_argument("--format", type=str, default="pth", choices=["pth", "safetensors"])
    p.add_argument("--lora_rank", type=int, default=16, help='LoRA rank used during training')
    p.add_argument(
        "--use_teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Export the teacher backbone instead of the student backbone',
    )

    pre, _ = p.parse_known_args()
    cfg_all = load_config_file(pre.config) if getattr(pre, "config", "") else {}
    cfg = select_section(cfg_all, "export_merge_lora_gigapath")

    cli = p.parse_args()
    merged = merge_config_and_cli(cfg, cli)

    if not merged.get("checkpoint"):
        p.error('Provide a checkpoint path.')
    if not merged.get("output"):
        p.error('Provide an output path.')

    if merged.get("use_teacher", None) is None:
        merged["use_teacher"] = False

    return argparse.Namespace(**merged)


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    key = "teacher" if args.use_teacher else "model"
    if key not in ckpt or "backbone" not in ckpt[key]:
        raise KeyError(f"Missing ckpt['{key}']['backbone'] in checkpoint {args.checkpoint}")
    model_sd = ckpt[key]["backbone"]

    inferred_ffn = infer_ffn_layer_from_state_dict(model_sd, default="swiglupacked")
    inferred_mlp_ratio = (
        infer_mlp_ratio_from_state_dict(model_sd)
        if inferred_ffn in ("mlp", "swiglupacked")
        else None
    )

    pe = model_sd.get("pos_embed", None) if isinstance(model_sd, dict) else None
    pos_embed_with_register_tokens = bool(isinstance(pe, torch.Tensor) and pe.ndim == 3 and pe.shape[1] > 197)

    backbone = build_gigapath_backbone(
        img_size=224,
        patch_size=16,
        num_register_tokens=0,
        init_values=1e-5,
        block_chunks=0,
        pos_embed_with_register_tokens=pos_embed_with_register_tokens,
        ffn_layer=inferred_ffn,
        mlp_ratio=inferred_mlp_ratio,
    )
    inject_lora_and_freeze(backbone, lora_rank=args.lora_rank, train_last_n_blocks=0, include_attn_key=True)
    backbone.load_state_dict(model_sd, strict=True)

    lora_merge_all(backbone, unmerge=False, key="lora_A")
    deactivate_lora(backbone, activate_layers=("attn",), delete_separate_proj=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "pth":
        torch.save(backbone.state_dict(), out_path)
    else:
        try:
            from safetensors.torch import save_file
        except Exception as e:
            raise RuntimeError('Safetensors export requires the safetensors package.') from e
        save_file(backbone.state_dict(), str(out_path))

    meta = {
        "architecture": "vit_giant_patch14_dinov2",
        "patch_size": 16,
        "embed_dim": 1536,
        "num_heads": 24,
        "depth": 40,
        "num_reg_tokens": 0,
        "mlp_ratio": float(inferred_mlp_ratio) if inferred_mlp_ratio is not None else 5.33334,
        "ffn_layer": inferred_ffn,
        "source": key,
    }
    (out_path.parent / (out_path.stem + ".meta.json")).write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"exported: {out_path}")


if __name__ == "__main__":
    main()
