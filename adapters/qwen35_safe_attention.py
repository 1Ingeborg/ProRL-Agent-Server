"""Qwen3.5 model spec with an exact Torch fallback for FLA L2 normalization.

The installed FLA/Triton combination can illegally access memory while
autotuning L2 normalization on SM80. The model and GDN rule remain unchanged.
"""
from functools import wraps
from types import MethodType
import torch
import torch.nn.functional as F
from slime_plugins.models import qwen3_5 as upstream


def safe_l2norm(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(dim=-1, keepdim=True) + 1e-6)).to(x.dtype)


def safe_chunk_rule(original):
    @wraps(original)
    def chunk(q, k, v, **kwargs):
        if kwargs.get("use_qk_l2norm_in_kernel", False):
            q, k = safe_l2norm(q), safe_l2norm(k)
            kwargs["use_qk_l2norm_in_kernel"] = False
        return original(q, k, v, **kwargs)
    return chunk


def safe_short_convolution(self, x, residual=None, mask=None, cache=None,
                           output_final_state=False, cu_seqlens=None, **kwargs):
    """Use cuDNN depthwise causal convolution, resetting at packed boundaries."""
    if cache is not None or output_final_state:
        raise ValueError("Training convolution does not use a recurrent cache")
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    boundaries = cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else [0, x.shape[1]]
    if cu_seqlens is not None and x.shape[0] != 1:
        raise ValueError("Packed convolution expects batch size 1")
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        part = x[:, start:end].transpose(1, 2).float().contiguous()
        y = F.conv1d(part, self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            padding=self.kernel_size[0]-1, groups=self.hidden_size)
        y = y[:, :, :end-start].transpose(1, 2)
        if self.activation in {"silu", "swish"}:
            y = F.silu(y)
        outputs.append(y.to(x.dtype))
    result = torch.cat(outputs, dim=1)
    if residual is not None:
        result = result + residual
    return result, None


class Qwen3_5GatedDeltaNet(upstream.Qwen3_5GatedDeltaNet):
    def __init__(self, config, layer_idx, args=None):
        super().__init__(config, layer_idx, args=args)
        if self.gdn_backend == "fla":
            self.chunk_gated_delta_rule = safe_chunk_rule(self.chunk_gated_delta_rule)
            self.conv1d.forward = MethodType(safe_short_convolution, self.conv1d)


def get_qwen3_5_spec(args, config, vp_stage):
    # Upstream Attention resolves this class when creating each linear layer.
    upstream.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNet
    return upstream.get_qwen3_5_spec(args, config, vp_stage)
