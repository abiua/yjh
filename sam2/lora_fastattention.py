import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# LoRA module with self-attention in the low-rank space
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoRAWithAttn(nn.Module):
    def __init__(self, dim, r=16, nhead=4, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, r, bias=False)
        self.up   = nn.Linear(r, dim, bias=False)
        # flash‐attn 不需要 batch_first
        self.to_q = nn.Linear(r, r, bias=False)
        self.to_k = nn.Linear(r, r, bias=False)
        self.to_v = nn.Linear(r, r, bias=False)
        self.norm = nn.LayerNorm(r)
        self.drop = nn.Dropout(dropout)
        self.nhead = nhead
        self.dropout = dropout

    def _attn3d(self, h):
        # h: [B, N, r], 其中 r = heads * head_dim
        B, N, r = h.shape
        head_dim = r // self.nhead

        # 1) 生成 Q/K/V 并拆成 heads
        #    → [B, N, heads, head_dim]
        q = self.to_q(h).reshape(B, N, self.nhead, head_dim)
        k = self.to_k(h).reshape(B, N, self.nhead, head_dim)
        v = self.to_v(h).reshape(B, N, self.nhead, head_dim)

        # 2) 切换到 [seq_len, batch, heads, head_dim]
        #    scaled_dot_product_attention 支持这种 shape
        q = q.permute(1, 0, 2, 3)  # [N, B, heads, head_dim]
        k = k.permute(1, 0, 2, 3)
        v = v.permute(1, 0, 2, 3)

        # 3) 调用 Flash‐Attention
        h2 = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout,
            is_causal=False
        )  # [N, B, heads, head_dim]

        # 4) 恢复回 [B, N, r]
        h2 = h2.permute(1, 0, 2, 3).reshape(B, N, r)

        # 5) 残差 + LayerNorm
        return self.norm(h + self.drop(h2))

    def forward(self, x):
        # 2D、3D、4D 都要先归一成 [B,N,C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B,N,C]
        elif x.dim() == 3:
            x_flat = x
        else:
            raise ValueError(...)
        # ↓ 下边都基于 x_flat
        h = self.down(x_flat)  # [B,N,r]
        h = self._attn3d(h)  # [B,N,r]
        out_flat = self.up(h)  # [B,N,C]
        # 再把 flat 还原回原始形状
        if x.dim() == 4:
            out = out_flat.permute(0, 2, 1).view(B, C, H, W)
        else:
            out = out_flat
        if x.dim() == 2:
            return out.squeeze(0)
        return out


class _LoRA_qkv_Attn(nn.Module):
    def __init__(self, orig_qkv, dim: int, r: int=16, nhead: int=4, dropout: float=0.1):
        super().__init__()
        self.orig = orig_qkv
        self.dim  = dim
        self.r    = r
        self.lora_q = LoRAWithAttn(dim, r, nhead, dropout)
        self.lora_v = LoRAWithAttn(dim, r, nhead, dropout)

    def forward(self, x):
        # 1) 先拿原始投影
        qkv = self.orig(x)   # Linear→[B,N,3C] 或 Conv2d→[B,3C,H,W]

        # 2) collapse 到 [B, N, 3*C]
        if qkv.dim() == 4 and qkv.shape[1] == self.dim*3:
            B, C3, H, W = qkv.shape
            qkv_flat = qkv.view(B, C3, H*W).permute(0,2,1)
        elif qkv.dim() == 4:  # NHWC
            B, H, W, C3 = qkv.shape
            qkv_flat = qkv.view(B, H*W, C3)
        elif qkv.dim() == 3:
            qkv_flat = qkv
        else:
            raise RuntimeError(f"Unsupported qkv output dim {qkv.shape}")

        # 3) split 出 Q/K/V
        q, k, v = qkv_flat.split(self.dim, dim=-1)  # 每个都是 [B, N, C]

        # 4) 在 Q/V 上加 LoRA‐Attn
        q = q + self.lora_q(q)
        v = v + self.lora_v(v)

        # 5) 拼回去
        out_flat = torch.cat([q, k, v], dim=-1)  # [B, N, 3C]

        # 6) 如果是 Conv2d 分支，再 reshape 回 [B,3C,H,W]
        if isinstance(self.orig, nn.Conv2d):
            return out_flat.permute(0,2,1).view(B, C3, H, W)
        return out_flat

# Main LoRA injector for SAM2
class LoRA_SAM2(nn.Module):
    def __init__(self, predictor, r: int=16, nhead: int=4, dropout: float=0.1, lora_layers=None):
        super().__init__()
        assert r > 0, "LoRA rank must be > 0"
        self.predictor = predictor.to(default_device)
        # determine which image encoder blocks get LoRA
        total_blocks = len(self.predictor.model.image_encoder.trunk.blocks)
        self.lora_layers = lora_layers or list(range(total_blocks))
        # freeze original image encoder
        for p in self.predictor.model.image_encoder.parameters():
            p.requires_grad = False
        # inject LoRA-attn into each specified block
        for idx, blk in enumerate(self.predictor.model.image_encoder.trunk.blocks):
            if idx not in self.lora_layers:
                continue
            # replace qkv linear
            orig = blk.attn.qkv
            if isinstance(orig, nn.Linear):
                # qkv.weight.shape = (3*C, in_features)，每路 Q/V 的通道数是 out_features // 3
                dim = orig.out_features // 3
            elif isinstance(orig, nn.Conv2d):
                # 同理：out_channels = 3*C
                dim = orig.out_channels // 3
            else:
                raise RuntimeError(f"Unsupported qkv type: {type(orig)}")
            blk.attn.qkv = _LoRA_qkv_Attn(
                orig_qkv = orig,
                dim = dim,
                r=r,
                nhead=nhead,
                dropout=dropout
            ).to(default_device)

    def forward(self, image):
        # identical logic to original predictor forward
        self.predictor.set_image_batch(image)
        sparse_emb, dense_emb = self.predictor.model.sam_prompt_encoder(
            points=None, boxes=None, masks=None)
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
            low_res, self.predictor._orig_hw[-1])
        return low_res, pred_masks, scores

    # you may keep your save/load functions here if needed
    # def save_lora_parameters(...): ...
    # def load_lora_parameters(...): ...

# Example usage:
# from your_sam_module import SamPredictor
# predictor = SamPredictor(...)  # initialize as usual
# lora_model = LoRA_SAM2(predictor, r=32, nhead=4)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=5e-4)
