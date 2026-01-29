import random

import pytest
import torch


from mojo_opset import MojoLightningIndexer
from mojo_opset.experimental import MojoIndexer
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented

TEST_SHAPES = [
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]
TEST_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


def precompute_freqs_cis(seqlen, dim, device) -> torch.Tensor:
    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs, device=device), freqs)
    return freqs_cis



