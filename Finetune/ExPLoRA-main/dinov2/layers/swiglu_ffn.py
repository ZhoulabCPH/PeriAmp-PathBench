# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import os
from typing import Callable, Optional
import warnings

from torch import Tensor, nn
import torch.nn.functional as F


class SwiGLUPacked(nn.Module):
    """Implement a timm-style packed SwiGLU FFN with fc1 and fc2 checkpoint keys. fc1 outputs hidden_features channels and fc2 consumes half as many. With a 1280-dimensional ViT-H and a ratio of 5.3375, fc1 outputs 6832 channels and fc2.weight has shape (1280, 3416)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if int(hidden_features) % 2 != 0:
            raise ValueError(f"SwiGLUPacked requires even hidden_features, got {hidden_features}")

        self.fc1 = nn.Linear(in_features, int(hidden_features), bias=bias)
        self.act = act_layer() if act_layer is not None else nn.SiLU()
        self.fc2 = nn.Linear(int(hidden_features) // 2, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if float(drop) > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.fc1(x)
        x1, x2 = x12.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (SwiGLU)")
    else:
        warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (SwiGLU)")


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
