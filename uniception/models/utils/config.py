"""
Model Utils Config
"""

import os
import warnings

import torch

__all__ = [
    "use_fused_attn",
    "set_fused_attn",
    "use_spas_sage2_attn",
    "set_spas_sage2_thresholds",
    "set_spas_sage2_enabled",
    "spas_sage2_attn_meansim_cuda",
    "SPAS_SIMTHRESHD1",
    "SPAS_CDFTHRESHD",
    "SPAS_PVTHRESHD",
    "_HAS_SPAS_SAGE2_ATTN",
]

# Use torch.scaled_dot_product_attention where possible
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, "scaled_dot_product_attention")
if "UNICEPTION_FUSED_ATTN" in os.environ:
    _USE_FUSED_ATTN = int(os.environ["UNICEPTION_FUSED_ATTN"])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on


def use_fused_attn() -> bool:
    "Return whether to use torch.nn.functional.scaled_dot_product_attention"
    return _USE_FUSED_ATTN > 0


def set_fused_attn(enable: bool = True):
    "Set whether to use torch.nn.functional.scaled_dot_product_attention"
    global _USE_FUSED_ATTN
    if not _HAS_FUSED_ATTN:
        warnings.warn("This version of pytorch does not have F.scaled_dot_product_attention, fused_attn flag ignored.")
        return
    if enable:
        _USE_FUSED_ATTN = 1
    else:
        _USE_FUSED_ATTN = 0


# Optional SPAS-SAGE2 attention kernel support and configuration
try:
    from spas_sage_attn import spas_sage2_attn_meansim_cuda  # type: ignore
    _HAS_SPAS_SAGE2_ATTN = True
except Exception:
    spas_sage2_attn_meansim_cuda = None  # type: ignore
    _HAS_SPAS_SAGE2_ATTN = False

# Thresholds (env-overridable) and runtime setters
SPAS_SIMTHRESHD1 = float(os.getenv("SPAS_SIMTHRESHD1", "0.6"))
SPAS_CDFTHRESHD = float(os.getenv("SPAS_CDFTHRESHD", "0.97"))
SPAS_PVTHRESHD = int(os.getenv("SPAS_PVTHRESHD", "15"))


def use_spas_sage2_attn() -> bool:
    "Return whether SPAS-SAGE2 attention is available and enabled"
    return _HAS_SPAS_SAGE2_ATTN


def set_spas_sage2_thresholds(simthreshd1: float, cdfthreshd: float, pvthreshd: int):
    "Set SPAS-SAGE2 attention thresholds"
    global SPAS_SIMTHRESHD1, SPAS_CDFTHRESHD, SPAS_PVTHRESHD
    SPAS_SIMTHRESHD1 = float(simthreshd1)
    SPAS_CDFTHRESHD = float(cdfthreshd)
    SPAS_PVTHRESHD = int(pvthreshd)


def set_spas_sage2_enabled(enabled: bool):
    "Enable or disable SPAS-SAGE2 attention"
    global _HAS_SPAS_SAGE2_ATTN
    _HAS_SPAS_SAGE2_ATTN = bool(enabled) and (spas_sage2_attn_meansim_cuda is not None)
