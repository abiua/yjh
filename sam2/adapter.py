# sam2/adapter.py
import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, dim: int, r: int = 16):
        super().__init__()
        self.down = nn.Linear(dim, r, bias=False)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dim = x.dim()
        if orig_dim == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            # —— 打印实际收到的 C 和期待的 dim ——
            print(f"Adapter got x_flat.shape = {x_flat.shape}, "
                  f"but expected dim = {self.down.weight.shape[1]}")
        elif orig_dim == 3:
            x_flat = x
        elif orig_dim == 2:
            x_flat = x.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        h = self.up(self.act(self.down(x_flat)))
        out_flat = x_flat + h

        if orig_dim == 4:
            out = out_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        elif orig_dim == 3:
            out = out_flat
        else:
            out = out_flat.squeeze(0)
        return out

