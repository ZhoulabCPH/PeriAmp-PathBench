"""
Merge LoRA into the Virchow2 backbone and export standalone weights.
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

from virchow2.modeling import (  # noqa: E402
    build_virchow2_backbone,
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
    p = argparse.ArgumentParser("Export merged Virchow2 backbone", add_help=True)
    p.add_argument(
        "--config",
        type=str,
        default=r"",
        help='JSON or YAML configuration. Command-line arguments take precedence.',
    )

    p.add_argument("--checkpoint", type=str, default=r"weights/virchow2_checkpoint.pth", help='Training checkpoint (.pth)')
    p.add_argument("--output", type=str, default="./outputs/virchow2_explora/e19_student/vit_huge_patch14_224_merged_e19.pth", help='Output path for the merged backbone weights (.pth or .safetensors)')
    p.add_argument("--format", type=str, default="pth", choices=["pth", "safetensors"])
    p.add_argument("--lora_rank", type=int, default=16, help='LoRA rank used during training')
    p.add_argument(
        "--use_teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Export the teacher backbone. The default exports the student backbone.',
    )

    pre, _ = p.parse_known_args()
    cfg_all = load_config_file(pre.config) if getattr(pre, "config", "") else {}
    cfg = select_section(cfg_all, "export_merge_lora")

    cli = p.parse_args()
    merged = merge_config_and_cli(cfg, cli)

    if not merged.get("checkpoint"):
        p.error('Provide --checkpoint or set checkpoint in the configuration.')
    if not merged.get("output"):
        p.error('Provide --output or set output in the configuration.')

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

    # Infer the FFN type from checkpoint keys and shapes before loading.
    inferred_ffn = infer_ffn_layer_from_state_dict(model_sd, default="swiglufused")
    inferred_mlp_ratio = infer_mlp_ratio_from_state_dict(model_sd) if inferred_ffn in ("mlp", "swiglupacked") else None

    pe = model_sd.get("pos_embed", None) if isinstance(model_sd, dict) else None
    pos_embed_with_register_tokens = bool(isinstance(pe, torch.Tensor) and pe.ndim == 3 and pe.shape[1] > 257)
    backbone = build_virchow2_backbone(
        img_size=224,
        patch_size=14,
        num_register_tokens=4,
        init_values=1e-5,
        block_chunks=0,
        pos_embed_with_register_tokens=pos_embed_with_register_tokens,
        ffn_layer=inferred_ffn,
        mlp_ratio=inferred_mlp_ratio,
    )
    # Rebuild the same backbone and LoRA branches used during training.
    inject_lora_and_freeze(backbone, lora_rank=args.lora_rank, train_last_n_blocks=0, include_attn_key=True)
    backbone.load_state_dict(model_sd, strict=True)

    # Merge the LoRA update into the linear weights: W <- W + BA.
    lora_merge_all(backbone, unmerge=False, key="lora_A")
    # Recombine Q, K, and V into QKV and remove the LoRA branches.
    deactivate_lora(backbone, activate_layers=("attn",), delete_separate_proj=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "pth":
        torch.save(backbone.state_dict(), out_path)
    else:
        try:
            from safetensors.torch import save_file
        except Exception as e:  # pragma: no cover
            raise RuntimeError('Safetensors export requires the safetensors package.') from e
        save_file(backbone.state_dict(), str(out_path))

    # Save a JSON metadata sidecar.
    meta = {
        "architecture": "vit_huge_patch14_224",
        "patch_size": 14,
        "embed_dim": 1280,
        "num_heads": 16,
        "depth": 32,
        "num_reg_tokens": 4,
        "mlp_ratio": float(inferred_mlp_ratio) if inferred_mlp_ratio is not None else 5.3375,
        "ffn_layer": inferred_ffn,
        "source": key,
    }
    (out_path.parent / (out_path.stem + ".meta.json")).write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


