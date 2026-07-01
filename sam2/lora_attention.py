import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# LoRA module with self-attention in the low-rank space
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoRAWithAttn(nn.Module):
    def __init__(self, dim, r=16, nhead=4, dropout=0.1):
        super().__init__()
        # down/up 低秩投影
        self.down = nn.Linear(dim, r, bias=False)
        self.up   = nn.Linear(r, dim, bias=False)
        # 低秩空间里的多头自注意力，注意不要用 batch_first=True
        self.to_q = nn.Linear(r, r, bias=False)
        self.to_k = nn.Linear(r, r, bias=False)
        self.to_v = nn.Linear(r, r, bias=False)
        self.norm = nn.LayerNorm(r)
        self.drop = nn.Dropout(dropout)
        self.nhead = nhead
        self.dropout = dropout

    def _attn3d(self, h):
        # h: [B, N, r]
        B, N, r = h.shape
        # 1) 生成 Q/K/V 并拆成 heads
        q = self.to_q(h).reshape(B, N, self.nhead, -1).permute(1, 0, 2, 3)  # [N, B, h, d]
        k = self.to_k(h).reshape(B, N, self.nhead, -1).permute(1, 0, 2, 3)
        v = self.to_v(h).reshape(B, N, self.nhead, -1).permute(1, 0, 2, 3)
        # 2) 调用 flash‐attn
        #    注意 scaled_dot_product_attention 接受 [L, B*heads, D] 或 [L, B, heads, D]
        h2 = F.scaled_dot_product_attention(
            q.flatten(2,3),         # [N, B*h, d]
            k.flatten(2,3),
            v.flatten(2,3),
            dropout_p=self.dropout,
            is_causal=False
        )
        # 3) 恢复成 [N, B, heads, d] → [B, N, r]
        h2 = h2.view(N, B, self.nhead, -1).permute(1, 0, 2, 3).reshape(B, N, r)
        # 4) 残差 + LN
        return self.norm(h + self.drop(h2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        支持 x 为 2D / 3D / 4D：
          - 2D: [N, dim] → 视为 (B=1) 张量
          - 3D: [B, N, dim]
          - 4D: [B, C, H, W]  (如 Conv2d 特征图)
        """
        # 2D → B=1
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # 4D：卷积特征 → flatten 空间维度
        if x.dim() == 4:
            B, C, H, W = x.shape
            # 把 [B,C,H,W] → [B, H*W, C]
            x_flat = x.reshape(B, C, H*W).permute(0, 2, 1)
        elif x.dim() == 3:
            # 已经是 [B, N, C]
            x_flat = x
        else:
            raise ValueError(f"LoRAWithAttn 不支持输入维度 {x.shape}")
            # <<< 在这里打 debug
        # print(f"LoRAWithAttn.forward: down.in_features={self.down.in_features}, "
        #       f"down.out_features={self.down.out_features}, "
        #       f"x_flat.shape[-1]={x_flat.shape[-1]}")

        # 低秩映射 + 注意力
        h = self.down(x_flat)                   # [B, N, r]
        h = self._attn3d(h)                     # [B, N, r]
        out_flat = self.up(h)                   # [B, N, dim]

        # 如果原来是 4D，就恢复形状
        if x.dim() == 4:
            out = out_flat.permute(0, 2, 1).view(B, C, H, W)
        else:
            out = out_flat

        # 如果原来是 2D，再 squeeze 回去
        if x.dim() == 2:
            return out.squeeze(0)
        return out


class _LoRA_qkv_Attn(nn.Module):
    def __init__(self, orig_qkv, dim: int, r: int=16, nhead: int=4, dropout: float=0.1):
        super().__init__()
        self.orig = orig_qkv  # 原始 qkv 层
        self.dim = dim  # 原始通道数 C
        self.r = r  # LoRA rank

        # 一定要写成 dim=self.dim, r=self.r
        self.lora_q = LoRAWithAttn(dim=self.dim, r=self.r, nhead=nhead, dropout=dropout)
        self.lora_v = LoRAWithAttn(dim=self.dim, r=self.r, nhead=nhead, dropout=dropout)

    def forward(self, x):
        # Call the original qkv layer on whatever it expects:
        qkv = self.orig(x)  # if Linear, x is already [B,N,C]; if Conv2d, x is [B,C,H,W]
        # 2) 根据输出维度统一折成 [B, N, 3*C]
        if qkv.dim() == 4:
            # 如果是 NCHW 排列 [B, 3C, H, W]
            if qkv.shape[1] == self.dim * 3:
                B, C3, H, W = qkv.shape
                qkv_flat = qkv.view(B, C3, H * W).permute(0, 2, 1)
            # 或者 NHWC 排列 [B, H, W, 3C]
            else:
                B, H, W, C3 = qkv.shape
                qkv_flat = qkv.view(B, H * W, C3)
        elif qkv.dim() == 3:
            # 已经是 [B, N, 3*C]
            qkv_flat = qkv
        else:
            raise RuntimeError(f"Unsupported qkv output dim: {qkv.shape}")

        # ——— 2) 现在 debug / assert
        # print(f"[DEBUG][_LoRA_qkv_Attn] qkv_flat.shape = {qkv_flat.shape}, expected dim*3 = {self.dim * 3}")
        assert qkv_flat.shape[-1] == self.dim * 3, (
            f"qkv_flat last dim mismatch: got {qkv_flat.shape[-1]}, but self.dim*3 = {self.dim * 3}"
        )

        # ——— 3) 切分 + LoRAWithAttn ……（不变）
        q, k, v = qkv_flat.split(self.dim, dim=-1)
        q = q + self.lora_q(q)
        v = v + self.lora_v(v)
        out_flat = torch.cat([q, k, v], dim=-1)

        # ——— 4) 恢复形状 + return
        if isinstance(self.orig, nn.Conv2d):
            return out_flat.permute(0, 2, 1).view(B, C3, H, W)
        else:
            return out_flat

        return out

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
