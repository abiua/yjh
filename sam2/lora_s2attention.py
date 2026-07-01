# sam2/lora_s2attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Space-to-2D Attention 的辅助函数 ---
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4]           = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:]   = x[:, :, 1:, 3 * c // 4:]
    return x

def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4]           = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:]   = x[:, 1:, :, 3 * c // 4:]
    return x

# --- Split-Attention ---
class SplitAttention(nn.Module):
    def __init__(self, channel, k=3):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_all):
        # x_all: [b, k, h, w, c]
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)            # [b, k, n, c]
        a = x_all.sum(dim=1).sum(dim=1)               # [b, c]
        hat = self.mlp2(self.gelu(self.mlp1(a)))      # [b, k*c]
        weights = hat.view(b, k, c)                   # [b, k, c]
        attn = self.softmax(weights).unsqueeze(2)     # [b, k, 1, c]
        out = (attn * x_all).sum(dim=1)               # [b, n, c]
        return out.view(b, h, w, c)                   # [b, h, w, c]

# --- S2Attention 核心实现 ---
class S2Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attn = SplitAttention(channel=channels, k=3)

    def forward(self, x):
        # x: [b, c, h, w] → [b, h, w, c]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)                             # [b, h, w, 3c]

        x1 = spatial_shift1(x[..., :c])              # [b, h, w, c]
        x2 = spatial_shift2(x[..., c:2*c])           # [b, h, w, c]
        x3 = x[..., 2*c:]                            # [b, h, w, c]

        x_all = torch.stack([x1, x2, x3], dim=1)     # [b, 3, h, w, c]
        a = self.split_attn(x_all)                   # [b, h, w, c]

        out = self.mlp2(a)                           # [b, h, w, c]
        return out.permute(0, 3, 1, 2)               # [b, c, h, w]

# --- LoRA-with-S2Attention ---
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoRAWithS2Attn(nn.Module):
    def __init__(self, dim, r=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.down = nn.Linear(dim, r, bias=False)
        self.up   = nn.Linear(r, dim, bias=False)
        self.s2att = S2Attention(channels=r)
        self.norm  = nn.LayerNorm(r)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        orig_dim = x.dim()
        if orig_dim == 2:
            x = x.unsqueeze(0)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        elif x.dim() == 3:
            x_flat = x                                   # [B, N, C]
        else:
            raise ValueError(f"Unsupported input dim {x.shape}")

        # 降维到 [B, N, r]
        h = self.down(x_flat)

        # 只有原始是 4D 时，才按空间映射回图、做 S2Attention
        if orig_dim == 4:
            h_map = h.permute(0, 2, 1).view(B, self.r, H, W)  # [B, r, H, W]
            h_map = self.s2att(h_map)                    # [B, r, H, W]
            h = h_map.view(B, self.r, H*W).permute(0, 2, 1)    # [B, N, r]

        # 残差 + LN
        h = self.norm(h + self.drop(h))

        # 升维回 [B, N, C]
        out_flat = self.up(h)

        # 恢复回输入形状
        if orig_dim == 4:
            out = out_flat.permute(0, 2, 1).view(B, C, H, W)
        else:
            out = out_flat
        if orig_dim == 2:
            out = out.squeeze(0)
        return out

class _LoRA_qkv_S2Attn(nn.Module):
    def __init__(self, orig_qkv, dim:int, r=16, dropout=0.1):
        super().__init__()
        self.orig   = orig_qkv
        self.dim    = dim
        self.r      = r
        self.lora_q = LoRAWithS2Attn(dim, r, dropout)
        self.lora_v = LoRAWithS2Attn(dim, r, dropout)

    def forward(self, x):
        # 1) 原 qkv 投影
        qkv = self.orig(x)

        # 2) collapse 到 [B, N, 3*C]
        if qkv.dim() == 4:
            B, C1, C2, C3 = qkv.shape
            # 如果是 NCHW 布局
            if C1 == self.dim * 3:
                # NCHW → [B, 3C, H, W]
                B, C3, H, W = qkv.shape
                qkv_flat = qkv.view(B, C3, H*W).permute(0, 2, 1)
            # 如果是 NHWC 布局
            elif C3 == self.dim * 3:
                # NHWC → [B, H, W, 3C]
                B, H, W, C3 = qkv.shape
                qkv_flat = qkv.view(B, H*W, C3)
            else:
                raise RuntimeError(f"Unsupported 4D qkv layout {qkv.shape}")
        elif qkv.dim() == 3:
            # [B, N, 3C]
            qkv_flat = qkv
        else:
            raise RuntimeError(f"Unsupported qkv output {qkv.shape}")

        # 3) split Q/K/V
        q, k, v = qkv_flat.split(self.dim, dim=-1)

        # 4) LoRA S2-attn
        q = q + self.lora_q(q)
        v = v + self.lora_v(v)

        # 5) 拼回 [B, N, 3C]
        out_flat = torch.cat([q, k, v], dim=-1)

        # 6) reshape 回原始布局
        if isinstance(self.orig, nn.Conv2d):
            # 如果原来是 NCHW
            if C1 == self.dim * 3:
                return out_flat.permute(0, 2, 1).view(B, C3, H, W)
            # 如果原来是 NHWC
            else:
                return out_flat.view(B, H, W, C3)
        else:
            return out_flat
