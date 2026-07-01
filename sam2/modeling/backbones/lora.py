# lora.py: 添加 LoRALinear
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: int = 16):
        super().__init__()
        # 冻结原始线性层参数
        self.weight = orig.weight
        self.bias = orig.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # LoRA 参数
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, orig.in_features))
        self.B = nn.Parameter(torch.zeros(orig.out_features, r))
        # 初始化
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = F.linear(x, self.weight, self.bias)
        # LoRA 增量
        lora_out = F.linear(x, self.B @ self.A, bias=None) * self.scaling
        return orig_out + lora_out