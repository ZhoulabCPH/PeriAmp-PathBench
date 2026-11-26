"""Shared DINO and iBOT heads, LoRA injection, EMA updates, and checkpoint inference utilities."""

import math
from typing import Dict, List, Optional, Tuple

import torch

from dinov2.layers import DINOHead
from dinov2.utils.train_lora_util import activate_lora


def build_dino_ibot_head(
    in_dim: int = 1280,
    out_dim: int = 65536,
    bottleneck_dim: int = 256,
    hidden_dim: int = 2048,
    nlayers: int = 3,
):
    return DINOHead(
        in_dim=in_dim,
        out_dim=out_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        nlayers=nlayers,
    )


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "backbone.",
        "student.backbone.",
        "teacher.backbone.",
    )
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _rename_known_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map common checkpoint parameter names, including names used by timm."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk == "reg_token":
            nk = "register_tokens"
        out[nk] = v
    return out


def infer_ffn_layer_from_state_dict(state_dict: Dict[str, torch.Tensor], default: str = "swiglufused") -> str:
    """Infer the FFN type from parameter names and shapes. fc1/fc2 can denote a standard MLP or timm SwiGLUPacked, while w12/w3 denotes fused SwiGLU."""
    if not isinstance(state_dict, dict) or not state_dict:
        return default
    keys = list(state_dict.keys())

    has_fc = any(".mlp.fc1." in k or ".mlp.fc2." in k for k in keys)
    has_w = any(".mlp.w12." in k or ".mlp.w3." in k for k in keys)
    if has_fc and not has_w:
        fc1_w_key = next((k for k in keys if k.endswith(".mlp.fc1.weight")), None)
        fc2_w_key = next((k for k in keys if k.endswith(".mlp.fc2.weight")), None)
        try:
            if fc1_w_key and fc2_w_key:
                fc1_w = state_dict.get(fc1_w_key)
                fc2_w = state_dict.get(fc2_w_key)
                if isinstance(fc1_w, torch.Tensor) and isinstance(fc2_w, torch.Tensor) and fc1_w.ndim == 2 and fc2_w.ndim == 2:
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


def infer_mlp_ratio_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[float]:
    """For a standard MLP, infer the ratio from fc2.weight as hidden_dim / embed_dim."""
    if not isinstance(state_dict, dict) or not state_dict:
        return None
    preferred = "blocks.0.mlp.fc2.weight"
    w = state_dict.get(preferred)
    if not isinstance(w, torch.Tensor):
        k = next((kk for kk in state_dict.keys() if kk.endswith(".mlp.fc2.weight")), None)
        w = state_dict.get(k) if k else None
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return None
    try:
        embed_dim = int(w.shape[0])
        hidden_dim = int(w.shape[1])
        if embed_dim <= 0 or hidden_dim <= 0:
            return None
        return float(hidden_dim) / float(embed_dim)
    except Exception:
        return None


def inject_lora_and_freeze(
    backbone: torch.nn.Module,
    lora_rank: int = 16,
    train_last_n_blocks: int = 2,
    include_attn_key: bool = True,
):
    """Inject LoRA into attention QKV layers, freeze the backbone, and train the adapters and the last train_last_n_blocks transformer blocks."""
    activate_lora(
        backbone,
        activate_layers=("attn",),
        lora_rank=lora_rank,
        include_attn_key=include_attn_key,
        include_attn_proj=False,
    )

    for p in backbone.parameters():
        p.requires_grad_(False)

    if hasattr(backbone, "mask_token") and isinstance(getattr(backbone, "mask_token"), torch.nn.Parameter):
        backbone.mask_token.requires_grad_(True)

    for name, p in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)

    if not hasattr(backbone, "blocks"):
        return
    blocks = backbone.blocks
    depth = len(blocks)
    start = max(depth - int(train_last_n_blocks), 0)
    for i in range(start, depth):
        for p in blocks[i].parameters():
            p.requires_grad_(True)


@torch.no_grad()
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, m: float):
    """
    Teacher EMA: theta_t = m * theta_t + (1-m) * theta_s
    """
    teacher_params = [p for p in teacher.parameters()]
    student_params = [p for p in student.parameters()]
    torch._foreach_mul_(teacher_params, m)
    torch._foreach_add_(teacher_params, student_params, alpha=1.0 - m)


def compute_ema_momentum(base_m: float, cur_step: int, max_steps: int) -> float:
    """Increase EMA momentum from base_m to 1.0 on a cosine schedule."""
    if max_steps <= 0:
        return 1.0
    t = min(max(cur_step / max_steps, 0.0), 1.0)
    t = 0.5 * (1.0 - math.cos(math.pi * t))
    return base_m + (1.0 - base_m) * t
