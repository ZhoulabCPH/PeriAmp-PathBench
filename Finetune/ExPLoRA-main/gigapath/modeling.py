"""Build GigaPath backbones, load PyTorch weights, and inject LoRA. The ViT-Giant tile encoder has width 1536, 40 blocks, 24 heads, 16-pixel patches, zero register tokens, and a packed SwiGLU FFN."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from dinov2.models import vision_transformer as vits

from common.modeling_utils import (
    build_dino_ibot_head,
    inject_lora_and_freeze,
    ema_update,
    compute_ema_momentum,
    _strip_known_prefixes,
    _rename_known_keys,
    infer_ffn_layer_from_state_dict,
    infer_mlp_ratio_from_state_dict,
)


__all__ = [
    "build_gigapath_backbone",
    "build_backbone_from_gigapath_config",
    "load_gigapath_weights",
    "infer_ffn_layer_from_pth",
    "infer_mlp_ratio_from_pth",
    "maybe_load_gigapath_config",
    "analyze_pth_compat",
    "build_dino_ibot_head",
    "inject_lora_and_freeze",
    "ema_update",
    "compute_ema_momentum",
    "infer_ffn_layer_from_state_dict",
    "infer_mlp_ratio_from_state_dict",
]


def build_gigapath_backbone(
    img_size: int = 224,
    patch_size: int = 16,
    num_register_tokens: int = 0,
    init_values: float = 1e-5,
    block_chunks: int = 0,
    pos_embed_with_register_tokens: bool = False,
    ffn_layer: str = "swiglupacked",
    mlp_ratio: Optional[float] = None,
):
    """Build the ViT-Giant backbone with GigaPath tile encoder defaults and a packed SwiGLU FFN."""
    backbone = vits.vit_giant_patch14_dinov2(
        img_size=img_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        pos_embed_with_register_tokens=pos_embed_with_register_tokens,
    )
    return backbone


def build_backbone_from_gigapath_config(
    config: dict,
    *,
    block_chunks: int = 0,
    ffn_layer: str = "swiglupacked",
    pos_embed_with_register_tokens: bool = False,
    mlp_ratio: Optional[float] = None,
):
    """Read model_args from GigaPath config.json, including the actual 16-pixel patch size used by vit_giant_patch14_dinov2."""
    arch = (config or {}).get("architecture", "vit_giant_patch14_dinov2")
    model_args = (config or {}).get("model_args", {}) or {}
    img_size = int(model_args.get("img_size", 224))
    patch_size = int(model_args.get("patch_size", 16))
    init_values = float(model_args.get("init_values", 1e-5))
    reg_tokens = int(model_args.get("reg_tokens", model_args.get("num_reg_tokens", 0)))

    if not hasattr(vits, arch):
        raise ValueError(
            f"Unknown architecture {arch!r}. No matching constructor in dinov2.models.vision_transformer."
        )
    builder = getattr(vits, arch)

    backbone = builder(
        img_size=img_size,
        patch_size=patch_size,
        num_register_tokens=reg_tokens,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        pos_embed_with_register_tokens=pos_embed_with_register_tokens,
    )
    return backbone


def load_gigapath_weights(
    backbone: torch.nn.Module,
    weights_path: str,
    strict: bool = False,
):
    """Load GigaPath .pth weights and normalize checkpoint prefixes. The backbone has zero register tokens and usually needs no positional embedding trimming."""
    obj = torch.load(weights_path, map_location="cpu")

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj
    elif isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "backbone"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
        else:
            sd = obj
    else:
        raise ValueError(f"Could not read the state dictionary from {weights_path}.")

    sd = _strip_known_prefixes(sd)
    sd = _rename_known_keys(sd)

    # Adjust positional embeddings if they differ from GigaPath's register-free layout.
    if "pos_embed" in sd:
        try:
            model_sd = backbone.state_dict()
            if "pos_embed" in model_sd:
                ckpt_pe = sd["pos_embed"]
                model_pe = model_sd["pos_embed"]
                if (
                    isinstance(ckpt_pe, torch.Tensor)
                    and isinstance(model_pe, torch.Tensor)
                    and ckpt_pe.ndim == 3
                    and model_pe.ndim == 3
                    and ckpt_pe.shape[0] == model_pe.shape[0] == 1
                    and ckpt_pe.shape[2] == model_pe.shape[2]
                    and ckpt_pe.shape[1] != model_pe.shape[1]
                ):
                    diff = int(ckpt_pe.shape[1] - model_pe.shape[1])
                    if diff > 0:
                        sd["pos_embed"] = torch.cat(
                            [ckpt_pe[:, :1, :], ckpt_pe[:, 1 + diff:, :]], dim=1
                        )
        except Exception:
            pass

    # Drop the optional classifier weights from the original GigaPath checkpoint.
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    missing, unexpected = backbone.load_state_dict(sd, strict=strict)
    return missing, unexpected


def infer_ffn_layer_from_pth(weights_path: str, default: str = "swiglupacked") -> str:
    """Infer the FFN type from a PyTorch checkpoint."""
    try:
        obj = torch.load(weights_path, map_location="cpu")
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        elif isinstance(obj, dict):
            for key in ("state_dict", "model", "model_state_dict", "backbone"):
                if key in obj and isinstance(obj[key], dict):
                    sd = obj[key]
                    break
            else:
                sd = obj
        else:
            return default
        return infer_ffn_layer_from_state_dict(sd, default=default)
    except Exception:
        return default


def infer_mlp_ratio_from_pth(weights_path: str) -> Optional[float]:
    """Infer the MLP ratio when a PyTorch checkpoint uses a standard MLP."""
    try:
        obj = torch.load(weights_path, map_location="cpu")
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        elif isinstance(obj, dict):
            for key in ("state_dict", "model", "model_state_dict", "backbone"):
                if key in obj and isinstance(obj[key], dict):
                    sd = obj[key]
                    break
            else:
                sd = obj
        else:
            return None
        return infer_mlp_ratio_from_state_dict(sd)
    except Exception:
        return None


def maybe_load_gigapath_config(config_json_path: Optional[str]) -> Optional[dict]:
    """Load the GigaPath config.json file when provided."""
    if not config_json_path:
        return None
    p = Path(config_json_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_pth_compat(
    backbone: torch.nn.Module,
    weights_path: str,
    *,
    topk: int = 30,
) -> Dict[str, object]:
    """Compare PyTorch checkpoint tensors with the current backbone state dictionary."""
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj
    elif isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "backbone"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
        else:
            sd = obj
    else:
        raise ValueError(f"Could not read the state dictionary from {weights_path}.")

    sd = _strip_known_prefixes(sd)
    sd = _rename_known_keys(sd)
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    if "pos_embed" in sd:
        try:
            model_sd = backbone.state_dict()
            if "pos_embed" in model_sd:
                ckpt_pe = sd["pos_embed"]
                model_pe = model_sd["pos_embed"]
                if (
                    isinstance(ckpt_pe, torch.Tensor)
                    and isinstance(model_pe, torch.Tensor)
                    and ckpt_pe.ndim == 3
                    and model_pe.ndim == 3
                    and ckpt_pe.shape[0] == model_pe.shape[0] == 1
                    and ckpt_pe.shape[2] == model_pe.shape[2]
                    and ckpt_pe.shape[1] != model_pe.shape[1]
                ):
                    diff = int(ckpt_pe.shape[1] - model_pe.shape[1])
                    if diff > 0:
                        sd["pos_embed"] = torch.cat(
                            [ckpt_pe[:, :1, :], ckpt_pe[:, 1 + diff:, :]], dim=1
                        )
        except Exception:
            pass

    model_sd = backbone.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(sd.keys())

    missing_keys: List[str] = sorted(list(model_keys - ckpt_keys))
    unexpected_keys: List[str] = sorted(list(ckpt_keys - model_keys))

    matched_keys: List[str] = []
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...], str, str]] = []
    for k in sorted(list(model_keys & ckpt_keys)):
        mv = model_sd[k]
        cv = sd[k]
        if not isinstance(mv, torch.Tensor) or not isinstance(cv, torch.Tensor):
            shape_mismatch.append((
                k,
                tuple(getattr(cv, "shape", ())),
                tuple(getattr(mv, "shape", ())),
                str(getattr(cv, "dtype", "")),
                str(getattr(mv, "dtype", "")),
            ))
            continue
        if tuple(cv.shape) == tuple(mv.shape):
            matched_keys.append(k)
        else:
            shape_mismatch.append((k, tuple(cv.shape), tuple(mv.shape), str(cv.dtype), str(mv.dtype)))

    summary = {
        "num_model_keys": len(model_keys),
        "num_ckpt_keys": len(ckpt_keys),
        "matched": len(matched_keys),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
        "shape_mismatch": len(shape_mismatch),
        "topk": int(topk),
    }
    return {
        "summary": summary,
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
    }
