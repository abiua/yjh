import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# -----------------------
# 1. ShuffleAttention 模块
# -----------------------
class ShuffleAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias   = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias   = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w).permute(0, 2, 1, 3, 4)
        return x.reshape(b, -1, h, w)

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # [b*G, c//G, h, w]
        x_0, x_1 = x.chunk(2, dim=1)      # each [b*G, c//(2G), h, w]

        # channel attention
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        # concat + reshuffle
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        return self.channel_shuffle(out, 2)


# -----------------------
# 2. LoRA-with-ShuffleAttention
# -----------------------
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoRAWithAttn(nn.Module):
    def __init__(self, dim, r=16, nhead=4, dropout=0.1, shuffle_groups=8):
        super().__init__()
        self.down = nn.Linear(dim, r, bias=False)
        self.up   = nn.Linear(r, dim, bias=False)
        # 插入 ShuffleAttention
        self.shuffle_attn = ShuffleAttention(channel=r, G=shuffle_groups)
        self.norm = nn.LayerNorm(r)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 支持 2D/3D/4D
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, N, C]
        elif x.dim() == 3:
            B, N, C = x.shape
            x_flat = x
            H = W = None
        else:
            raise ValueError(f"Unsupported dim {x.shape}")

        # down 投影到低秩空间
        h = self.down(x_flat)  # [B, N, r]
        # 还原到 [B, r, H, W] 用于 ShuffleAttention
        if x.dim() == 4:
            h_conv = h.permute(0, 2, 1).view(B, -1, H, W)
            h_conv = self.shuffle_attn(h_conv)
            h = h_conv.view(B, -1, H*W).permute(0, 2, 1)
        # 3D 输入则跳过 ShuffleAttention
        # h: [B, N, r]

        # 残差 + LN
        h = self.norm(h + self.drop(h))

        # up 投影回原始维度
        out_flat = self.up(h)  # [B, N, C]

        # 恢复成 4D / 3D / 2D
        if x.dim() == 4:
            out = out_flat.permute(0, 2, 1).view(B, C, H, W)
        else:
            out = out_flat
        if x.dim() == 2:
            out = out.squeeze(0)
        return out


class _LoRA_qkv_Attn(nn.Module):
    def __init__(self, orig_qkv, dim: int, r: int=16, nhead: int=4, dropout: float=0.1, shuffle_groups=8):
        super().__init__()
        self.orig   = orig_qkv
        self.dim    = dim
        self.r      = r
        self.lora_q = LoRAWithAttn(dim, r, nhead, dropout, shuffle_groups)
        self.lora_v = LoRAWithAttn(dim, r, nhead, dropout, shuffle_groups)

    def forward(self, x):
        # 1) 原始 QKV 投影
        qkv = self.orig(x)  # Linear→[B,N,3C] 或 Conv2d→[B,3C,H,W]

        # 2) collapse 到 [B, N, 3*C]
        if qkv.dim() == 4 and qkv.shape[1] == self.dim*3:
            B, C3, H, W = qkv.shape
            qkv_flat = qkv.view(B, C3, H*W).permute(0,2,1)
        elif qkv.dim() == 4:
            B, H, W, C3 = qkv.shape
            qkv_flat = qkv.view(B, H*W, C3)
        elif qkv.dim() == 3:
            qkv_flat = qkv
        else:
            raise RuntimeError(f"Unsupported qkv dim {qkv.shape}")

        # 3) 切分 Q/K/V
        q, k, v = qkv_flat.split(self.dim, dim=-1)

        # 4) 只在 Q/V 上加 LoRAWithAttn
        q = q + self.lora_q(q)
        v = v + self.lora_v(v)

        # 5) 拼回去
        out_flat = torch.cat([q, k, v], dim=-1)

        # 6) 如果是 Conv2d，再 reshape 回 [B,3C,H,W]
        if isinstance(self.orig, nn.Conv2d):
            return out_flat.permute(0,2,1).view(B, C3, H, W)
        return out_flat


# -----------------------
# 3. LoRA-SAM2 注入器
# -----------------------
class LoRA_SAM2(nn.Module):
    def __init__(self, predictor, r=16, nhead=4, dropout=0.1, lora_layers=None, shuffle_groups=8):
        super().__init__()
        self.predictor = predictor.to(default_device)
        total_blocks = len(self.predictor.model.image_encoder.trunk.blocks)
        self.lora_layers = lora_layers or list(range(total_blocks))

        # 冻结原始 encoder
        for p in self.predictor.model.image_encoder.parameters():
            p.requires_grad = False

        # 注入 LoRA-QKV-Attn
        for idx, blk in enumerate(self.predictor.model.image_encoder.trunk.blocks):
            if idx not in self.lora_layers:
                continue
            orig = blk.attn.qkv
            if isinstance(orig, nn.Linear):
                dim = orig.out_features // 3
            elif isinstance(orig, nn.Conv2d):
                dim = orig.out_channels // 3
            else:
                raise RuntimeError(f"Unsupported qkv type {type(orig)}")
            blk.attn.qkv = _LoRA_qkv_Attn(
                orig_qkv=orig,
                dim=dim,
                r=r,
                nhead=nhead,
                dropout=dropout,
                shuffle_groups=shuffle_groups
            ).to(default_device)

    def forward(self, image):
        # 与原 predictor 一致的前向逻辑
        self.predictor.set_image_batch(image)
        sparse_emb, dense_emb = self.predictor.model.sam_prompt_encoder(
            points=None, boxes=None, masks=None
        )
        hr_feats = [f for f in self.predictor._features["high_res_feats"]]
        low_res, scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"],
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
            repeat_image=True,
            high_res_features=hr_feats,
        )
        pred_masks = self.predictor._transforms.postprocess_masks(
            low_res, self.predictor._orig_hw[-1]
        )
        return low_res, pred_masks, scores

# —— END ——
