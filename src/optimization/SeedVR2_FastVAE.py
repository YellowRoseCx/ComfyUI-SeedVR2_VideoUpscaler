"""SeedVR2 Fast VAE support — drop-in patch for single frame upscaling.
"""

import logging
import math
import os

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Single-frame fast path for the video VAE (3-D conv -> 2-D conv when T=1)
# --------------------------------------------------------------------------------------
# Why this is exact, not an approximation:
#
# InflatedCausalConv3d moves its temporal padding out of the conv (self.padding[0] = 0)
# and instead replicates frame 0 `2 * temporal_padding` times ahead of the input.
# For a single-frame input that yields:
#
#     2 * temporal_padding + 1  ==  kt      identical copies of frame 0
#
# and a Conv3d over exactly `kt` frames with zero temporal padding produces T_out = 1 for
# ANY temporal stride, because floor((kt - kt) / st) + 1 == 1. Every temporal tap therefore
# sees the same frame, so
#
#     out = sum_t  W[:, :, t] * frame0   ==   conv2d(frame0, sum_t W[:, :, t])
#
# The saving is both compute (kt-1 of every kt multiply-accumulates were redundant) and
# peak memory (the kt-replicated input tensor is never built).

def apply_fast_vae_patch(enable=True):
    import torch
    import torch.nn.functional as F
    import math
    import logging
    from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
    from ..models.video_vae_v3.modules.types import MemoryState

    log = logging.getLogger(__name__)

    if not enable:
        # Revert if patched
        if getattr(InflatedCausalConv3d.forward, "_seedvr2_1_4b", False):
            InflatedCausalConv3d.forward = InflatedCausalConv3d.forward._seedvr2_orig
            log.info("SeedVR2-FastVAE: Disabled Fast VAE Patch.")
        return False

    orig_forward = InflatedCausalConv3d.forward
    if getattr(orig_forward, "_seedvr2_1_4b", False):
        return True # Already patched

    def _collapsed_forward(self, input, memory_state):
        """The T=1 path. Assumes the caller has already verified the guards."""
        if memory_state != MemoryState.ACTIVE:
            self.memory = None

        kt = self.kernel_size[0]
        frame0 = input[:, :, :1]

        if memory_state != MemoryState.DISABLED:
            if math.isinf(self.memory_limit):
                n = kt - self.stride[0]
            else:
                n = kt - self.stride[0]
                if memory_state not in (MemoryState.INITIALIZING, MemoryState.ACTIVE):
                    n = 0
            if n > 0:
                self.memory = frame0.expand(-1, -1, n, -1, -1).detach().contiguous()
            elif math.isinf(self.memory_limit):
                self.memory = None

        weight = self.weight
        bias = self.bias

        try:
            # Sum the temporal taps in fp32 so a 3-term fp16 accumulation can't lose bits,
            # then return to the conv's working dtype.
            w2d = weight.float().sum(dim=2).to(weight.dtype)
            out = F.conv2d(
                frame0[:, :, 0],
                w2d,
                bias,
                stride=self.stride[1:],
                padding=self.padding[1:],       # padding[0] is 0 by construction
                dilation=self.dilation[1:],
                groups=self.groups,
            )
        except Exception as e:
            raise e

        return out.unsqueeze(2)

    def forward(self, input, memory_state=MemoryState.UNSET):
        if (
            memory_state != MemoryState.UNSET
            and torch.is_tensor(input)
            and input.ndim == 5
            and input.shape[2] == 1                    # single frame
            and self.kernel_size[0] > 1                # kt == 1 has nothing to collapse
            and self.dilation[0] == 1
            and self.temporal_padding * 2 == self.kernel_size[0] - 1
            and not (
                memory_state == MemoryState.ACTIVE
                and self.memory is not None
            )
        ):
            try:
                return _collapsed_forward(self, input, memory_state)
            except Exception as e:
                log.debug("SeedVR2-FastVAE: 2-D collapse fell back to 3-D", exc_info=True)

        return orig_forward(self, input, memory_state)

    forward._seedvr2_1_4b = True
    forward._seedvr2_orig = orig_forward
    InflatedCausalConv3d.forward = forward
    log.info("SeedVR2-FastVAE: Enabled single-frame Fast VAE Patch.")
    return True
