"""Build and load Virchow2 ViT-H/14 backbones. The model uses width 1280, 32 blocks, 16 heads, 14-pixel patches, four register tokens, fused SwiGLU, and safetensors weights."""

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
    "build_virchow2_backbone",
    "build_backbone_from_virchow2_config",
    "load_virchow2_weights",
    "infer_ffn_layer_from_safetensors",
    "infer_mlp_ratio_from_safetensors",
    "maybe_load_virchow2_config",
    "analyze_safetensors_compat",
    "build_dino_ibot_head",
    "inject_lora_and_freeze",
    "ema_update",
    "compute_ema_momentum",
    "infer_ffn_layer_from_state_dict",
    "infer_mlp_ratio_from_state_dict",
]


def build_virchow2_backbone(
    img_size: int = 224,
    patch_size: int = 14,
    num_register_tokens: int = 4,
    init_values: float = 1e-5,
    block_chunks: int = 0,
    pos_embed_with_register_tokens: bool = False,
    ffn_layer: str = "swiglufused",
    mlp_ratio: Optional[float] = None,
):
    """Build a Virchow2 ViT-H/14 backbone. Official weights usually use fused SwiGLU, and the FFN type and MLP ratio can be overridden to match other checkpoints."""
    backbone = vits.vit_huge_patch14_224(
        img_size=img_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        pos_embed_with_register_tokens=pos_embed_with_register_tokens,
    )
    return backbone


def build_backbone_from_virchow2_config(
    config: dict,
    *,
    block_chunks: int = 0,
    ffn_layer: str = "swiglufused",
    pos_embed_with_register_tokens: bool = True,
    mlp_ratio: Optional[float] = None,
):
    """Build the backbone from config.json and match the checkpoint's positional embedding and register token layout."""
    arch = (config or {}).get("architecture", "vit_huge_patch14_224")
    model_args = (config or {}).get("model_args", {}) or {}
    img_size = int(model_args.get("img_size", 224))
    init_values = float(model_args.get("init_values", 1e-5))
    reg_tokens = int(model_args.get("reg_tokens", model_args.get("num_reg_tokens", 0)))

    if not hasattr(vits, arch):
        raise ValueError(f"Unknown architecture {arch!r}. No matching constructor in dinov2.models.vision_transformer.")
    builder = getattr(vits, arch)

    backbone = builder(
        img_size=img_size,
        num_register_tokens=reg_tokens,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        pos_embed_with_register_tokens=pos_embed_with_register_tokens,
    )
    return backbone


def infer_ffn_layer_from_safetensors(safetensors_path: str, default: str = "swiglufused") -> str:
    """Infer the FFN type from parameter names in a safetensors checkpoint."""
    try:
        from safetensors import safe_open
    except Exception:
        return default

    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return default

    has_fc = any(".mlp.fc1." in k or ".mlp.fc2." in k for k in keys)
    has_w = any(".mlp.w12." in k or ".mlp.w3." in k for k in keys)
    if has_fc and not has_w:
        try:
            fc1_w_key = next((k for k in keys if k.endswith(".mlp.fc1.weight")), None)
            fc2_w_key = next((k for k in keys if k.endswith(".mlp.fc2.weight")), None)
            if fc1_w_key and fc2_w_key:
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    fc1_w = f.get_tensor(fc1_w_key)
                    fc2_w = f.get_tensor(fc2_w_key)
                if getattr(fc1_w, "ndim", 0) == 2 and getattr(fc2_w, "ndim", 0) == 2:
                    fc1_out = int(fc1_w.shape[0])
                    fc2_in = int(fc2_w.shape[1])
                    if fc1_out == fc2_in:
                        return "mlp"
                    if fc1_out == 2 * fc2_in:
                        return "swiglupacked"
        except Exception:
            pass
        return "mlp"
    if has_w and not has_fc:
        return "swiglufused"
    return default


def infer_mlp_ratio_from_safetensors(safetensors_path: str) -> Optional[float]:
    """Infer the standard MLP ratio from the shape of fc2.weight in a safetensors checkpoint."""
    try:
        from safetensors import safe_open
    except Exception:
        return None

    preferred = "blocks.0.mlp.fc2.weight"
    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if preferred in keys:
                k = preferred
            else:
                k = next((kk for kk in keys if kk.endswith(".mlp.fc2.weight")), None)
                if k is None:
                    return None
            w = f.get_tensor(k)
    except Exception:
        return None

    try:
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            return None
        embed_dim = int(w.shape[0])
        hidden_dim = int(w.shape[1])
        if embed_dim <= 0 or hidden_dim <= 0:
            return None
        return float(hidden_dim) / float(embed_dim)
    except Exception:
        return None


def load_virchow2_weights(backbone: torch.nn.Module, safetensors_path: str, strict: bool = False):
    """Load Virchow2 weights from safetensors and normalize common parameter prefixes."""
    try:
        from safetensors.torch import load_file as safe_load
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Install safetensors to load these weights.') from e

    sd = safe_load(safetensors_path)
    sd = _strip_known_prefixes(sd)
    sd = _rename_known_keys(sd)

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
                    if diff > 0 and ckpt_pe.shape[1] == model_pe.shape[1] + diff:
                        sd["pos_embed"] = torch.cat([ckpt_pe[:, :1, :], ckpt_pe[:, 1 + diff:, :]], dim=1)
        except Exception:
            pass
    missing, unexpected = backbone.load_state_dict(sd, strict=strict)
    return missing, unexpected


def analyze_safetensors_compat(
    backbone: torch.nn.Module,
    safetensors_path: str,
    *,
    topk: int = 30,
) -> Dict[str, object]:
    """Compare safetensors weights with the backbone state dictionary."""
    try:
        from safetensors.torch import load_file as safe_load
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Install safetensors to inspect these weights.') from e

    sd = safe_load(safetensors_path)
    sd = _strip_known_prefixes(sd)
    sd = _rename_known_keys(sd)

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
                    if diff > 0 and ckpt_pe.shape[1] == model_pe.shape[1] + diff:
                        sd["pos_embed"] = torch.cat([ckpt_pe[:, :1, :], ckpt_pe[:, 1 + diff:, :]], dim=1)
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
            shape_mismatch.append((k, tuple(getattr(cv, "shape", ())), tuple(getattr(mv, "shape", ())), str(getattr(cv, "dtype", "")), str(getattr(mv, "dtype", ""))))
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


def maybe_load_virchow2_config(config_json_path: Optional[str]) -> Optional[dict]:
    if not config_json_path:
        return None
    p = Path(config_json_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
