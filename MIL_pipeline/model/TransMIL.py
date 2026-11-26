import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(
        self,
        dim=512,
        num_heads=8,
        dim_head=None,
        num_landmarks=None,
        pinv_iterations=6,
        attn_dropout=0.1,
        residual=True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        dim_head = dim_head or max(dim // num_heads, 1)
        num_landmarks = num_landmarks or max(dim // 2, 1)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim_head,
            heads=num_heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            residual=residual,
            dropout=attn_dropout
        )

    def forward(self, x, mask=None):
        return x + self.attn(self.norm(x), mask=mask)


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(
        self,
        n_classes=2,
        in_dim=512,
        hidden_dim=512,
        num_heads=8,
        dim_head=None,
        num_landmarks=None,
        pinv_iterations=6,
        attn_dropout=0.1,
        fc_dropout=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=hidden_dim)

        fc_blocks = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        if fc_dropout > 0:
            fc_blocks.append(nn.Dropout(fc_dropout))
        self._fc1 = nn.Sequential(*fc_blocks)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.n_classes = n_classes
        self.layer1 = TransLayer(
            dim=hidden_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            attn_dropout=attn_dropout,
            norm_layer=norm_layer,
        )
        self.layer2 = TransLayer(
            dim=hidden_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            attn_dropout=attn_dropout,
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, self.n_classes)

    def forward(self, h, mask=None):
        device = h.device
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)

        h = self._fc1(h)

        if mask is not None:
            h = h * mask.unsqueeze(-1)

        B, N, _ = h.shape
        grid_size = int(np.ceil(np.sqrt(N)))
        target_len = grid_size * grid_size
        add_length = target_len - N

        if add_length > 0:
            pad_tokens = h[:, :add_length, :]
            h = torch.cat([h, pad_tokens], dim=1)
            if mask is not None:
                pad_mask = mask[:, :add_length]
                mask = torch.cat([mask, pad_mask], dim=1)

        _H = grid_size
        _W = grid_size

        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        attn_mask = None
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
            mask = torch.cat([cls_mask, mask], dim=1)
            attn_mask = mask

        h = self.layer1(h, mask=attn_mask)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h, mask=attn_mask)

        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        return logits


if __name__ == "__main__":
    model = TransMIL()
    data = torch.rand((1, 1000, 512))
    mask = torch.ones((1, 1000), dtype=torch.bool)
    out = model(data, mask)
    print(out)