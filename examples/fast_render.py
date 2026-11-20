"""Batched time-varying renderer: same maths as ctract's Renderer, one pass.

``Renderer.forward`` loops over ``batch * blocks`` in Python and builds every
block's response separately. For a whole utterance (200+ blocks) that is the
bottleneck: 0.22 s of forward+backward per 16 ms block. Every block's response
is independent, so they can be evaluated as one batch instead -- the network is
solved for all blocks at once, the IRs are one tensor, and the excitation is
convolved block-by-block in a single matmul-shaped FFT.

Verified against ``Renderer.forward`` on a stationary and on a moving geometry
(relative difference < 1e-12 in float64).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ctract_chimera.renderer import Renderer, sample_frames


def render_blocks(renderer: Renderer, geometry: dict[str, Tensor], flow: Tensor,
                  frame_size: int, temperature: float | Tensor | None = 1e-5) -> Tensor:
    """(1,L) flow, geometry (1,T,sections) -> (L,) audio, all blocks batched."""
    if flow.ndim != 2 or flow.shape[0] != 1:
        raise ValueError("flow must be (1,L)")
    bs = renderer.block_size
    length = flow.shape[-1]
    blocks = (length + bs - 1) // bs
    positions = (torch.arange(blocks, dtype=flow.dtype) * bs + (bs - 1) / 2) / frame_size
    keys = ("ph", "orl", "ph_len", "or_len", "velum")
    values = [sample_frames(geometry[key], positions).flatten(0, 1) for key in keys]

    el = renderer.acoustics.elements(*values, temperature=temperature)
    nodes, dipoles = renderer.acoustics.discrete_responses(el, renderer.freqs, temperature)
    fraction = renderer.freqs / renderer.sample_rate
    correction = torch.exp(1j * torch.pi * fraction * (1 - renderer.sample_rate / renderer.acoustics.fs))
    correction = correction * torch.sinc(renderer.freqs / renderer.acoustics.fs) / torch.sinc(fraction)
    nodes = nodes * correction[None, :, None]
    response = nodes[..., renderer.acoustics.inject]                     # (blocks, F)

    full_ir = torch.fft.irfft(response, n=renderer.response_fft)        # (blocks, R)
    ir = torch.roll(full_ir, renderer.pre_response, -1)[..., : renderer.ir_length]

    fft_size = 1 << (bs + renderer.ir_length - 2).bit_length()
    src = torch.nn.functional.pad(flow, (0, blocks * bs - length)).reshape(blocks, bs)
    convolved = torch.fft.irfft(torch.fft.rfft(src, n=fft_size) * torch.fft.rfft(ir, n=fft_size),
                                n=fft_size)[..., : bs + renderer.ir_length - 1]

    out_length = blocks * bs + renderer.ir_length - 1
    out = torch.zeros(out_length, dtype=convolved.dtype)
    for i in range(blocks):                                             # overlap-add, vectorised by rows
        out[i * bs: i * bs + convolved.shape[-1]] += convolved[i]
    out = (out - torch.nn.functional.pad(out[:-1], (1, 0))) * renderer.sample_rate
    return out[renderer.pre_response: renderer.pre_response + length]
