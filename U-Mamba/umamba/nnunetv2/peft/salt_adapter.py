import torch, torch.nn as nn, torch.nn.functional as F
from typing import Union

class SALTAdapter(nn.Module):
    """
    Drop-in replacement for nn.Conv{2,3}d or nn.Linear that keeps *exactly*
    the same spatial behaviour (stride, padding, dilation, groups).
    """
    def __init__(self,
                 layer: Union[nn.Conv2d, nn.Conv3d, nn.Linear],
                 rank_svd: int = 32,
                 rank_lora: int = 4):
        super().__init__()
        self.is_conv = isinstance(layer, (nn.Conv2d, nn.Conv3d))

        # ── keep geometry for convs ──────────────────────────────────────────
        if self.is_conv:
            self.stride   = layer.stride
            self.padding  = layer.padding
            self.dilation = layer.dilation
            self.groups   = layer.groups
            self.orig_shape = layer.weight.shape            # (out, in, k, k[, k])
            weight_2d = layer.weight.reshape(layer.out_channels, -1)
        else:                                               # Linear
            self.orig_shape = layer.weight.shape            # (out, in)
            weight_2d = layer.weight

        # ── SVD in float32 on CPU, then move buffers back ───────────────────
        device, dtype = layer.weight.device, layer.weight.dtype
        U, S, Vt = torch.linalg.svd(weight_2d.cpu(), full_matrices=False)
        top_r = min(rank_svd, S.size(0))

        self.register_buffer("U",      U[:, :top_r].to(device, dtype))
        self.register_buffer("Vt",     Vt[:top_r, :].to(device, dtype))
        self.register_buffer("S_orig", S[:top_r]   .to(device, dtype))

        # trainable SALT parameters
        self.scale  = nn.Parameter(torch.ones(top_r,              device=device, dtype=dtype))
        self.shift  = nn.Parameter(torch.zeros(top_r,             device=device, dtype=dtype))
        self.lora_A = nn.Parameter(torch.randn(rank_lora, Vt.size(1),
                                               device=device, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(self.orig_shape[0], rank_lora,
                                               device=device, dtype=dtype) * 0.02)

        self.bias = (layer.bias.to(device) if layer.bias is not None else None)

    def _recompose_weight(self):
        # (the code you already have that builds W_hat)
        scales = self.scale * self.S_orig + self.shift          # (k,)
        W_svd  = (self.U * scales.unsqueeze(0)) @ self.Vt
        W_lora = (self.lora_B @ self.lora_A) / self.lora_A.size(0)
        W_hat  = W_svd + W_lora
        return W_hat.view(self.orig_shape) if self.is_conv else W_hat

    # ─────────────────────────────────────────────────────────────────────
    def weight(self):
        """
        Return *the same* W_hat tensor each time it is queried during one
        forward pass (= prevents duplicate graphs). Cleared automatically
        after backward via a hook.
        """
        if getattr(self, "_cached_weight", None) is None:
            self._cached_weight = self._recompose_weight()

            # clear cache after backward so next iteration recomputes
            def _clear_cache(_):
                self._cached_weight = None
            self._cached_weight.register_hook(_clear_cache)

        return self._cached_weight
    weight = property(weight)     # keep the @property interface

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, x):
        W = self.weight
        if self.is_conv:
            if   x.dim() == 4:     # 2-D
                return F.conv2d(x, W, self.bias,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups)
            elif x.dim() == 5:     # 3-D
                return F.conv3d(x, W, self.bias,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups)
            else:
                raise ValueError(f"SALTAdapter got conv weight but x.dim()={x.dim()}")
        else:                      # Linear
            return F.linear(x, W, self.bias)

        

    def merged_weight(self):
        """Helper for exporting an *inference-only* checkpoint."""
        return self.weight.detach().clone()
    
    def weight_detached(self):
        return self.weight.detach()

def apply_salt(model: nn.Module, rank_svd=32, rank_lora=4):
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            wrapped = SALTAdapter(module, rank_svd, rank_lora)
            wrapped = wrapped.to(module.weight.device, module.weight.dtype)
            setattr(model, name, wrapped)
        else:
            apply_salt(module, rank_svd, rank_lora)
    for p in model.parameters():
        p.requires_grad = isinstance(p, nn.Parameter) and p.grad_fn is None

# def apply_salt(model: nn.Module,
#                rank_svd: int = 32,
#                rank_lora: int = 4,
#                target_is: tuple = (nn.Conv2d, nn.Conv3d, nn.Linear)):
#     """
#     Recursively replaces layers in `target_is` with SALTAdapter,
#     freezes everything else.
#     """
#     for name, module in list(model.named_children()):
#         if isinstance(module, target_is):
#             wrapped = SALTAdapter(module.weight, module.bias,
#                                   rank_svd, rank_lora)
#             wrapped = wrapped.to(module.weight.device)
#             setattr(model, name, wrapped)
#         else:
#             apply_salt(module, rank_svd, rank_lora, target_is)

#     # freeze original params so only SALTAdapter is trained
#     for p in model.parameters():
#         p.requires_grad = isinstance(p, nn.Parameter) and p.grad_fn is None