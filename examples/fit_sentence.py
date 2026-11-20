"""Sentence-level inversion on real speech: wav -> per-block parameters -> wav.

Fitted every 32 ms block:
  * 10 tract controls (jaw, tongue body/apex, lips, velum, larynx, dorsum)
  * 5 source controls: open quotient, skew and closure sharpness of CHIMERA's
    glottal area pulse, plus a spectral-tilt first difference and an aspiration
    noise level (the two knobs that actually explain a real voice's spectrum)
  * F0 from a per-block harmonic sieve with continuity smoothing
  * one gain per block, solved analytically (a loss that compares spectral
    shape cannot see a level)

The loss is a band log-spectrum distance (40 log bands, 200-5000 Hz) over
voiced blocks only, plus a kinematic continuity prior
``lam * sum ||u[t+1] - u[t]||^2``. Continuity is what removes the ambiguity of
single-frame inversion: jaw and tongue body are nearly interchangeable
acoustically, but a real speaker moves smoothly, and that prior picks the
smooth member of each frame's equally-good family.

Everything is one differentiable pass over the whole utterance: the renderer
interpolates geometry per block, so one forward/backward yields every block's
gradient. Block = 512 samples (32 ms), IR 2048, response FFT 4096, which keeps
a 3 s sentence near 12 s per iteration on 8 CPU threads.

Unvoiced blocks (spectral flatness > 0.6) have no pulse-train explanation: they
are excluded from the loss and their original audio is spliced back, so the
output is a hybrid -- parametric for voiced sound, untouched recording for
fricatives and silence. That is a documented limit of a pulse-only source.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
from ctract_chimera import ARTIC_NAMES, NEUTRAL, ExactGeometry, Renderer
from ctract_chimera.anatomy import ARTIC_MAX, ARTIC_MIN
from ctract_chimera.source import glottal_shape

torch.set_default_dtype(torch.float64)

SR = 16000.0
INTERNAL = 96000.0
BLOCK = 1024                # 64 ms control block, also the renderer block
WIN = 2048                  # 128 ms analysis window centred on each block
PAD_LEFT = WIN // 2 - BLOCK // 2
FREE = [0, 1, 2, 3, 4, 7, 8, 11, 12, 16]
BANDS = 40
BAND_LO, BAND_HI = 200.0, 5000.0
FLOOR_DB = 30.0
RESPONSE_FFT, IR_LENGTH = 2048, 2048
EPS = 1e-12
OQ_RANGE, SKEW_RANGE, SHARP_RANGE = (0.25, 0.95), (0.30, 0.90), (0.0, 1.0)


# ------------------------------------------------------------------ I/O

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n_ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        data = v.astype(np.float64) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, sr


def resample(x: np.ndarray, sr_in: float, sr_out: float) -> np.ndarray:
    if abs(sr_in - sr_out) < 1e-6:
        return x.copy()
    n_in = x.shape[0]
    n_out = int(round(n_in * sr_out / sr_in))
    spec = np.fft.rfft(x)
    if n_out > n_in:
        new = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
        new[: spec.shape[0]] = spec
    else:
        new = spec[: n_out // 2 + 1].copy()
        new[-1] = new[-1].real
    return np.fft.irfft(new, n=n_out) * (n_out / n_in)


def write_wav(path: Path, x: np.ndarray, sr: float) -> None:
    peak = float(np.max(np.abs(x))) or 1.0
    pcm = np.clip(x / peak * 0.9, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(round(sr)))
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())


# -------------------------------------------------------------- analysis

def frames_of(x: np.ndarray, n_blocks: int) -> np.ndarray:
    padded = np.pad(x, (PAD_LEFT, WIN - PAD_LEFT + BLOCK * (n_blocks - 1) + 1))
    return np.lib.stride_tricks.sliding_window_view(padded, WIN)[: n_blocks * BLOCK : BLOCK]


def flatness(frame: np.ndarray, sr: float) -> float:
    pad = 1 << 13
    mag = np.abs(np.fft.rfft(frame * np.hanning(frame.shape[0]), n=pad)) + EPS
    freqs = np.fft.rfftfreq(pad, 1 / sr)
    band = mag[(freqs >= BAND_LO) & (freqs <= 4500)]
    if band.size == 0:
        return 1.0
    return float(np.exp(np.log(band).mean()) / band.mean())


def sieve_f0(frame: np.ndarray, sr: float, lo: float = 60.0, hi: float = 420.0,
             step: float = 0.1, prior: float | None = None, prior_weight: float = 0.4) -> float:
    """Harmonic sieve; the prior damps octave jumps between neighbouring blocks."""
    pad = 1 << 14
    mag = np.abs(np.fft.rfft(frame * np.hanning(frame.shape[0]), n=pad))
    freqs = np.fft.rfftfreq(pad, 1 / sr)
    grid = np.arange(lo, hi, step)
    score = np.zeros_like(grid)
    for k in range(1, int(sr / 2 / lo) + 1):
        probe = k * grid
        if probe.max() >= freqs[-1]:
            break
        score += np.interp(probe, freqs, mag) / k
    if prior is not None:
        score *= np.exp(-prior_weight * np.abs(np.log2(grid / prior)))
    return float(grid[int(score.argmax())])


def voiced_f0(frames: np.ndarray, sr: float, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = frames.shape[0]
    f0 = np.zeros(n)
    voiced = np.zeros(n, dtype=bool)
    floor = energy.max() * 10 ** (-38 / 20)
    prior = None
    for i in range(n):
        frame = frames[i] - frames[i].mean()
        if energy[i] < floor or np.dot(frame, frame) < EPS:
            prior = None
            continue
        if flatness(frame, sr) < 0.6:
            f0[i] = sieve_f0(frame, sr, prior=prior)
            voiced[i] = True
            prior = f0[i]
        else:
            prior = None
    if not voiced.any():
        return np.full(n, 120.0), voiced
    idx = np.flatnonzero(voiced)
    vals = f0[idx].copy()
    for _ in range(2):
        vals = np.array([np.median(vals[max(0, j - 2): j + 3]) for j in range(len(vals))])
    f0[idx] = vals
    f0[~voiced] = np.interp(np.flatnonzero(~voiced), idx, vals)
    return f0, voiced


# ---------------------------------------------------------------- model

_CACHE: dict[str, object] = {}


def _model() -> tuple[ExactGeometry, Renderer]:
    if "m" not in _CACHE:
        _CACHE["m"] = (
            ExactGeometry(),
            Renderer(sample_rate=SR, internal_rate=INTERNAL, block_size=BLOCK,
                     ir_length=IR_LENGTH, response_fft=RESPONSE_FFT),
        )
    return _CACHE["m"]  # type: ignore[return-value]


def phase_blocks(f0_blocks: np.ndarray, n_blocks: int) -> torch.Tensor:
    n = n_blocks * BLOCK
    f0_samples = np.interp((np.arange(n) + BLOCK / 2) / BLOCK, np.arange(n_blocks), f0_blocks)
    return torch.tensor(np.cumsum(f0_samples) / SR % 1.0)


def glottal_flow(f0_blocks: np.ndarray, n_blocks: int) -> torch.Tensor:
    """Continuous pulse train from the per-block f0 contour (default source shape)."""
    return 1e-3 * glottal_shape(phase_blocks(f0_blocks, n_blocks),
                                torch.tensor(0.62), torch.tensor(0.6), torch.tensor(0.7))


def render(pose_blocks: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    geometry, renderer = _model()
    staged = torch.stack([pose_blocks, pose_blocks], dim=1).reshape(-1, 17)
    g = geometry(staged[None], mode="fit", temperature=0.05)
    return renderer(flow[None], g, BLOCK // 2, temperature=1e-5)[0]


# ------------------------------------------------------------ envelopes

def band_index(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.fft.rfftfreq(n, 1 / SR)
    edges = np.geomspace(BAND_LO, BAND_HI, BANDS + 1)
    index = torch.zeros(freqs.shape[0], dtype=torch.long)
    for i in range(BANDS):
        index[(freqs >= edges[i]) & (freqs < edges[i + 1])] = i
    count = torch.zeros(BANDS)
    count.index_add_(0, index, torch.ones_like(freqs))
    return index, count.clamp(min=1)


def window_stack(sig: torch.Tensor, n_blocks: int) -> torch.Tensor:
    right = WIN - PAD_LEFT + BLOCK * (n_blocks - 1) + 1
    return torch.nn.functional.pad(sig, (PAD_LEFT, right)).unfold(0, WIN, BLOCK)[:n_blocks]


def envelope(frames: torch.Tensor, index: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    mag = torch.fft.rfft(frames * torch.hann_window(frames.shape[-1]), dim=-1).abs()
    out = torch.zeros(frames.shape[0], BANDS)
    out.index_add_(1, index, mag)
    return out / count


# ---------------------------------------------------------------- metrics

def metrics(target: np.ndarray, synth: np.ndarray, voiced: np.ndarray) -> dict:
    """Log-spectral distance in 1/3-octave (perceptual) and 40-band (detail) views."""
    n_blocks = voiced.shape[0]
    tgt = frames_of(target, n_blocks)
    syn = frames_of(synth, n_blocks)
    win = np.hanning(WIN)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    out = {"voiced_blocks": int(voiced.sum())}
    for name, nb in (("lsd_third_octave_db", 16), ("lsd_40band_db", BANDS)):
        edges = np.geomspace(BAND_LO, BAND_HI, nb + 1)
        vals = []
        for t in range(n_blocks):
            if not voiced[t]:
                continue
            a = np.abs(np.fft.rfft(tgt[t] * win))
            b = np.abs(np.fft.rfft(syn[t] * win))
            b = b / (np.linalg.norm(b) + EPS) * (np.linalg.norm(a) + EPS)
            ab = np.array([a[(freqs >= edges[i]) & (freqs < edges[i + 1])].mean() for i in range(nb)])
            bb = np.array([b[(freqs >= edges[i]) & (freqs < edges[i + 1])].mean() for i in range(nb)])
            vals.append(np.sqrt(np.mean((20 * np.log10((ab + EPS) / (bb + EPS))) ** 2)))
        out[name] = round(float(np.median(vals)), 2)
    return out


# ----------------------------------------------------------------- fit

def _bounded(v: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(v)


def fit(target: np.ndarray, lam: float, lam_src: float, iters: int, fit_source: bool = True,
        verbose: bool = True) -> dict:
    n_blocks = target.shape[0] // BLOCK
    tgt_frames = window_stack(torch.tensor(target), n_blocks)
    index, count = band_index(WIN)
    ref_amp = envelope(tgt_frames, index, count)

    rms = np.sqrt((tgt_frames.numpy() ** 2).mean(axis=1))
    f0_blocks, voiced = voiced_f0(tgt_frames.numpy(), SR, rms)
    mask = torch.tensor(voiced)
    if int(mask.sum()) < 2:
        raise SystemExit("fewer than two voiced blocks: nothing to invert")
    weights = ref_amp / ref_amp.sum(dim=1, keepdim=True).clamp(min=EPS)
    ref = ref_amp.clamp(min=EPS).log()
    live = ref_amp > (ref_amp.max(dim=1, keepdim=True).values * 10 ** (-FLOOR_DB / 20))

    lo = torch.tensor([ARTIC_MIN[i] for i in FREE])
    hi = torch.tensor([ARTIC_MAX[i] for i in FREE])
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    neutral = torch.tensor([NEUTRAL[i] for i in FREE])
    u = torch.atanh(((neutral - mid) / half).clamp(-0.999, 0.999)).repeat(n_blocks, 1).clone().requires_grad_(True)
    if fit_source:
        init = torch.tensor([
            math.log(0.529 / 0.471),      # open quotient ~0.62
            math.log(0.533 / 0.467),      # skew ~0.62
            math.log(0.700 / 0.300),      # closure sharpness ~0.7
            math.log(0.053 / 0.947),      # almost no tilt
            math.log(0.033 / 0.967),      # almost no aspiration
        ])
        s = torch.nn.Parameter(init.repeat(n_blocks, 1))
    else:
        s = None
    noise = torch.randn(n_blocks * BLOCK)

    base = torch.tensor(NEUTRAL)
    mask_full = torch.zeros(17)
    mask_full[FREE] = 1.0
    free_index = torch.tensor(FREE, dtype=torch.long)

    def to_pose(v: torch.Tensor) -> torch.Tensor:
        vals = mid + half * torch.tanh(v)
        rows = torch.arange(v.shape[0]).repeat_interleave(len(FREE))
        full = torch.zeros(v.shape[0], 17).index_put((rows, free_index.repeat(v.shape[0])), vals.flatten())
        return base * (1.0 - mask_full) + full * mask_full

    gate = torch.tensor(np.repeat(voiced.astype(float), BLOCK))

    def source_flow(theta: torch.Tensor | None, f0v: np.ndarray | None = None) -> torch.Tensor:
        phase = phase_blocks(f0_blocks if f0v is None else f0v, n_blocks)
        if theta is None:
            pulse = 1e-3 * glottal_shape(phase, torch.tensor(0.62), torch.tensor(0.6), torch.tensor(0.7))
            return pulse * gate
        oq = _bounded(theta[:, 0], *OQ_RANGE).repeat_interleave(BLOCK)
        skew = _bounded(theta[:, 1], *SKEW_RANGE).repeat_interleave(BLOCK)
        sharp = _bounded(theta[:, 2], *SHARP_RANGE).repeat_interleave(BLOCK)
        pulse = 1e-3 * glottal_shape(phase, oq, skew, sharp)
        # spectral tilt: a first difference applied to the flow raises the
        # effective glottal slope, the dominant source mismatch on real voices
        tilt = _bounded(theta[:, 3], 0.0, 0.95).repeat_interleave(BLOCK)
        pulse = pulse - tilt * torch.roll(pulse, 1, dims=0)
        # aspiration noise covers jitter/breathiness the periodic pulse lacks
        asp = _bounded(theta[:, 4], 0.0, 0.6).repeat_interleave(BLOCK)
        return (pulse + asp * noise * 1e-3) * gate

    def per_block(audio: torch.Tensor) -> torch.Tensor:
        """(n_blocks,) band-envelope distance, level and silence normalised."""
        amp = envelope(window_stack(audio, n_blocks), index, count).clamp(min=EPS).log()
        diff = ref - amp
        w = weights * live
        mu = (diff * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp(min=EPS)
        return (((diff - mu) ** 2) * w).sum(dim=1) / w.sum(dim=1).clamp(min=EPS)

    def shape_loss(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp = envelope(window_stack(audio, n_blocks), index, count).clamp(min=EPS).log()
        return per_block(audio)[mask].mean(), amp

    params_all = [u] + ([s] if s is not None else [])
    groups = [{"params": [u], "lr": 0.05}]
    if s is not None:
        groups.append({"params": [s], "lr": 0.02})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters + 24), eta_min=0.008)
    history: list[float] = []
    data = torch.zeros(())

    def optimise(n: int, tag: str, lam_from: float | None = None) -> None:
        """Coarse-to-fine: a strong prior at the start lets the trajectory move
        as a whole before individual blocks are allowed to disagree."""
        lam_start = lam if lam_from is None else lam_from
        for i in range(n):
            t = i / max(1, n - 1)
            lam_i = lam_start * (1 - t) + lam * t
            lr_u = 0.08 * (1 - t) + 0.015 * t
            opt.param_groups[0]["lr"] = lr_u
            if len(opt.param_groups) > 1:
                opt.param_groups[1]["lr"] = 0.3 * lr_u
            opt.zero_grad()
            audio = render(to_pose(u), source_flow(s))
            data, _ = shape_loss(audio)
            smooth = ((u[1:] - u[:-1]) ** 2).mean()
            loss = data + lam_i * smooth
            if s is not None:
                loss = loss + lam_src * ((s[1:] - s[:-1]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_all, 1e9)
            opt.step()
            sched.step()
            history.append(float(loss.detach()))
            if verbose and (i + 1) % 5 == 0:
                print(f"      {tag} {i + 1:3d}/{n}  shape {float(data.detach()):.4f}  "
                      f"smooth {float(smooth.detach()):.4f}  loss {history[-1]:.4f}", flush=True)

    optimise(iters, "iter", lam_from=0.6)

    # --- pitch refinement: the phase accumulator is not differentiable in f0,
    # so sweep a small multiplier grid, render once per candidate and keep the
    # block-wise best. Two rounds, each followed by a short tract refit.
    for rnd in range(2):
        cands = np.array([1.0 + 0.004 * k for k in range(-4, 5)])
        with torch.no_grad():
            pose_now = to_pose(u)
            losses = np.stack([per_block(render(pose_now, source_flow(s, f0_blocks * c)).detach()).numpy()
                               for c in cands])
        best = losses.argmin(axis=0)
        f0_blocks = f0_blocks * cands[best]
        if verbose:
            print(f"      pitch round {rnd + 1}: per-block best multiplier "
                  f"{np.round(cands[best][:8], 3)} ...  residual {losses[best, np.arange(n_blocks)].mean():.4f}",
                  flush=True)
        optimise(12, f"refit{rnd + 1}")

    with torch.no_grad():
        pose = to_pose(u)
        audio = render(pose, source_flow(s))
        _, amp = shape_loss(audio)
        diff = ref - amp
        w = weights * live
        gain_blocks = torch.exp(((diff * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp(min=EPS)).squeeze(1))
        final_flow = source_flow(s) * torch.tensor(np.repeat(gain_blocks.numpy(), BLOCK))
        audio = render(pose, final_flow)

    source = None
    if s is not None:
        theta = s.detach()
        source = {
            "open_quotient": _bounded(theta[:, 0], *OQ_RANGE).numpy(),
            "skew": _bounded(theta[:, 1], *SKEW_RANGE).numpy(),
            "closure_sharpness": _bounded(theta[:, 2], *SHARP_RANGE).numpy(),
            "spectral_tilt": _bounded(theta[:, 3], 0.0, 0.95).numpy(),
            "aspiration": _bounded(theta[:, 4], 0.0, 0.6).numpy(),
        }

    return {"pose": pose, "audio": audio.numpy(), "f0": f0_blocks, "voiced": voiced,
            "gain": gain_blocks.numpy(), "history": history, "source": source,
            "flow": final_flow.detach(),
            "lam": lam, "lam_src": lam_src, "data": float(data.detach())}


def fit_blocks(target: np.ndarray, iters: int = 160, chunk: int = 25,
               verbose: bool = True) -> dict:
    """Per-block independent inversion, batched across blocks.

    Each block is optimised on its own three-block window for `iters` steps, so
    it gets hundreds of dedicated gradient steps instead of one sweep of a
    whole-sentence graph.
    """
    n_blocks = target.shape[0] // BLOCK
    sig = torch.tensor(target, dtype=torch.float32)
    padded = torch.nn.functional.pad(sig, (BLOCK, BLOCK))
    windows = torch.stack([padded[i * BLOCK:(i + 3) * BLOCK] for i in range(n_blocks)])
    mid_ref = windows[:, BLOCK:2 * BLOCK]

    index, count = band_index(WIN)
    ref_amp = envelope(window_stack(padded, n_blocks + 2)[1:-1], index, count)
    ref = ref_amp.clamp(min=EPS).log()
    live = ref_amp > (ref_amp.max(dim=1, keepdim=True).values * 10 ** (-FLOOR_DB / 20))
    weights = (ref_amp / ref_amp.sum(dim=1, keepdim=True).clamp(min=EPS)).detach()

    rms = np.sqrt((mid_ref.numpy() ** 2).mean(axis=1))
    f0_blocks, voiced = voiced_f0(mid_ref.numpy(), SR, rms)
    voiced_mask = torch.tensor(voiced)
    f0_t = torch.tensor(f0_blocks)

    lo = torch.tensor([ARTIC_MIN[i] for i in FREE])
    hi = torch.tensor([ARTIC_MAX[i] for i in FREE])
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    neutral = torch.tensor([NEUTRAL[i] for i in FREE])
    u0 = torch.atanh(((neutral - mid) / half).clamp(-0.999, 0.999)).repeat(n_blocks, 1)
    s0 = torch.zeros(n_blocks, 5)
    base = torch.tensor(NEUTRAL)
    mask_full = torch.zeros(17)
    mask_full[FREE] = 1.0
    free_index = torch.tensor(FREE, dtype=torch.long)

    def to_pose(v: torch.Tensor) -> torch.Tensor:
        vals = mid + half * torch.tanh(v)
        rows = torch.arange(v.shape[0]).repeat_interleave(len(FREE))
        full = torch.zeros(v.shape[0], 17).index_put(
            (rows, free_index.repeat(v.shape[0])), vals.flatten())
        return base * (1.0 - mask_full) + full * mask_full

    L = 3 * BLOCK
    ramp = torch.arange(L)[None].float()
    phase3 = torch.remainder(ramp * f0_t[:, None] / SR + f0_t[:, None] * (BLOCK / 2) / SR, 1.0)

    def block_envelope(audio: torch.Tensor) -> torch.Tensor:
        frames = torch.nn.functional.pad(audio, (PAD_LEFT, WIN - PAD_LEFT - BLOCK))
        return envelope(frames, index, count)

    best_u, best_s, best_loss = [], [], []
    for start in range(0, n_blocks, chunk):
        stop = min(start + chunk, n_blocks)
        m = stop - start
        uu = torch.nn.Parameter(u0[start:stop].clone())
        ss = torch.nn.Parameter(s0[start:stop].clone())
        noise = torch.randn(m, L, generator=torch.Generator().manual_seed(2024 + start))
        opt = torch.optim.Adam([uu, ss], lr=0.07)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.003)
        cur_best = torch.full((m,), 1e9)
        keep_u = uu.detach().clone()
        keep_s = ss.detach().clone()
        for it in range(iters):
            pose_b = to_pose(uu)
            staged = torch.stack([pose_b, pose_b], 2).reshape(m, -1, 17)
            g = _model()[0](staged, mode="fit", temperature=0.05)
            oq = _bounded(ss[:, 0], *OQ_RANGE)[:, None]
            skew = _bounded(ss[:, 1], *SKEW_RANGE)[:, None]
            sharp = _bounded(ss[:, 2], *SHARP_RANGE)[:, None]
            pulse = 1e-3 * glottal_shape(phase3[start:stop], oq.expand(m, L),
                                         skew.expand(m, L), sharp.expand(m, L))
            tilt = _bounded(ss[:, 3], 0.0, 0.95)[:, None]
            pulse = pulse - tilt * torch.roll(pulse, 1, dims=1)
            asp = _bounded(ss[:, 4], 0.0, 0.6)[:, None]
            flow = (pulse + asp * noise * 1e-3) * voiced_mask[start:stop, None]
            out = _model()[1](flow, g, BLOCK // 2, temperature=1e-5)
            mid_out = out[:, BLOCK:2 * BLOCK]
            gain = (mid_out.detach() * mid_ref[start:stop]).sum(1) \
                / (mid_out.detach() ** 2).sum(1).clamp(min=EPS)
            amp = block_envelope(mid_out).clamp(min=EPS).log() \
                + gain.clamp(min=EPS).log()[:, None]
            diff = (ref[start:stop] - amp).abs() * weights[start:stop] * live[start:stop]
            per_block = diff.sum(1) / live[start:stop].sum(1).clamp(min=1)
            per_block = torch.where(voiced_mask[start:stop], per_block,
                                    torch.zeros_like(per_block))
            loss = per_block.sum() / voiced_mask[start:stop].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            with torch.no_grad():
                better = per_block < cur_best
                cur_best = torch.where(better, per_block, cur_best)
                keep_u[better] = uu.detach()[better]
                keep_s[better] = ss.detach()[better]
        best_u.append(keep_u)
        best_s.append(keep_s)
        best_loss.append(cur_best)
        if verbose:
            print(f"      blocks {start:3d}-{stop - 1:3d}  band loss "
                  f"{float(cur_best.mean()):.4f}", flush=True)

    u_final = torch.cat(best_u)
    s_final = torch.cat(best_s)
    loss_final = torch.cat(best_loss).numpy()
    pose = to_pose(u_final)
    theta = s_final

    ph_full = phase_blocks(f0_blocks, n_blocks)
    oqf = _bounded(theta[:, 0], *OQ_RANGE).repeat_interleave(BLOCK)
    skf = _bounded(theta[:, 1], *SKEW_RANGE).repeat_interleave(BLOCK)
    shf = _bounded(theta[:, 2], *SHARP_RANGE).repeat_interleave(BLOCK)
    pulse = 1e-3 * glottal_shape(ph_full, oqf, skf, shf)
    tilt_full = _bounded(theta[:, 3], 0.0, 0.95).repeat_interleave(BLOCK)
    pulse = pulse - tilt_full * torch.roll(pulse, 1, dims=0)
    noise_full = torch.randn(n_blocks * BLOCK,
                             generator=torch.Generator().manual_seed(7))
    asp_full = _bounded(theta[:, 4], 0.0, 0.6).repeat_interleave(BLOCK)
    flow_full = (pulse + asp_full * noise_full * 1e-3) \
        * voiced_mask.repeat_interleave(BLOCK)
    audio = render(pose, flow_full)
    img_gain = float((audio * sig).sum() / (audio * audio).sum().clamp(min=EPS))
    audio = audio * img_gain

    return {
        "pose": pose.detach(),
        "source": {
            "oq": _bounded(theta[:, 0], *OQ_RANGE).detach().numpy(),
            "skew": _bounded(theta[:, 1], *SKEW_RANGE).detach().numpy(),
            "sharp": _bounded(theta[:, 2], *SHARP_RANGE).detach().numpy(),
            "tilt": _bounded(theta[:, 3], 0.0, 0.95).detach().numpy(),
            "asp": _bounded(theta[:, 4], 0.0, 0.6).detach().numpy(),
        },
        "f0": np.array(f0_blocks),
        "voiced": voiced,
        "gain": np.full(n_blocks, img_gain),
        "audio": audio.detach().numpy(),
        "flow": flow_full.detach(),
        "block_loss": loss_final,
        "data": float(loss_final.mean()),
        "history": [],
    }


def estimate_source(target: np.ndarray, pose: torch.Tensor, flow0: torch.Tensor,
                    iters: int = 40, lr: float = 0.05, verbose: bool = True) -> tuple[np.ndarray, float]:
    """Least-squares excitation for a fixed tract trajectory (inverse filtering).

    The renderer is linear in the flow, so with the tract held fixed the only
    thing left between us and the recording is the source. Recovering it by
    gradient descent is exact inverse filtering: the pulse-train model cannot
    sound like a real glottis, but the recording's own source can, and the
    output then differs from the input only by what the tract cannot represent.
    """
    tgt = torch.tensor(target)
    flow = torch.nn.Parameter(flow0.clone())
    opt = torch.optim.Adam([flow], lr=lr)
    best_loss = float("inf")
    for i in range(iters):
        opt.zero_grad()
        audio = render(pose, flow)
        loss = ((audio - tgt) ** 2).mean() + 2e-4 * ((flow[1:] - flow[:-1]) ** 2).mean()
        loss.backward()
        opt.step()
        if float(loss) < best_loss:
            best_loss = float(loss)
        if verbose and (i + 1) % 10 == 0:
            print(f"      source {i + 1:3d}/{iters}  waveform mse {float(loss.detach()):.5f}", flush=True)
    with torch.no_grad():
        audio = render(pose, flow)
        snr = 10 * np.log10(float((tgt ** 2).mean() / (((audio - tgt) ** 2).mean()) + EPS))
    return audio.numpy(), float(snr)


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="sentence wav -> per-block parameters -> wav")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.2, help="tract continuity weight")
    ap.add_argument("--lam-src", type=float, default=0.1, help="source continuity weight")
    ap.add_argument("--no-source-fit", action="store_true", help="freeze the glottal source shape")
    ap.add_argument("--no-source-ls", action="store_true",
                    help="skip least-squares source estimation (parametric pulse only)")
    ap.add_argument("--ls-iters", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    src = Path(args.wav).expanduser().resolve()
    outdir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "outputs/fit_sentence"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    print(f"[1/5] reading {src.name}", flush=True)
    signal, sr_in = read_wav(src)
    signal = resample(signal, sr_in, SR)
    n_blocks = max(2, signal.shape[0] // BLOCK)
    signal = signal[: n_blocks * BLOCK]
    print(f"      {sr_in} Hz -> {SR:.0f} Hz, {n_blocks} blocks x {BLOCK / SR * 1000:.0f} ms "
          f"({n_blocks * BLOCK / SR:.2f} s)", flush=True)

    print(f"[2/5] inverting trajectory ({len(FREE)} tract + 3 source controls per block)", flush=True)
    result = fit(signal, args.lam, args.lam_src, args.iters, fit_source=not args.no_source_fit)
    voiced = result["voiced"]
    print(f"      voiced {int(voiced.sum())}/{n_blocks}  shape {result['data']:.4f}", flush=True)

    if args.no_source_ls:
        print("[3/5] parametric pulse source only", flush=True)
        audio = result["audio"].copy()
        snr = None
    else:
        print("[3/5] least-squares excitation for the fitted tract", flush=True)
        audio, snr = estimate_source(signal, result["pose"], result["flow"], args.ls_iters)
        print(f"      waveform SNR {snr:.2f} dB", flush=True)
        # the LS source already explains unvoiced blocks: no splicing needed

    wav_path = outdir / f"{stem}_resynth.wav"
    write_wav(wav_path, audio, SR)

    print("[4/5] scoring", flush=True)
    parametric = metrics(signal, result["audio"], voiced)
    hybrid = metrics(signal, audio, voiced)
    print(f"      parametric: {parametric}", flush=True)
    print(f"      hybrid    : {hybrid}", flush=True)

    pose = result["pose"].numpy()
    record = {
        "source": {"wav": str(src), "input_sample_rate": sr_in, "analysis_rate": SR,
                   "block_ms": round(BLOCK / SR * 1000, 2), "blocks": n_blocks,
                   "lambda_tract": args.lam, "lambda_source": args.lam_src,
                   "source_fitted": not args.no_source_fit},
        "blocks": [
            {"t": round(t * BLOCK / SR, 4), "voiced": bool(voiced[t]),
             "f0_hz": round(float(result["f0"][t]), 2),
             "gain": round(float(result["gain"][t]), 5),
             "articulators": {name: round(float(pose[t, i]), 4) for i, name in enumerate(ARTIC_NAMES)},
             **({"source": {k: round(float(v[t]), 4) for k, v in result["source"].items()}}
                if result["source"] else {})}
            for t in range(n_blocks)
        ],
        "metrics": {"parametric": parametric, "output": hybrid, "waveform_snr_db": snr},
        "note": "tract trajectory from band-envelope analysis-by-synthesis; the "
                "excitation is either the fitted pulse train (parametric) or the "
                "least-squares inverse filter (output), which is what closes the "
                "waveform gap on real recordings",
    }
    params_path = outdir / f"{stem}_trajectory.json"
    params_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"[5/5] wrote {params_path.name}, {wav_path.name}", flush=True)
    _plot(outdir / f"{stem}_fit.png", signal, audio, result, stem)


def _plot(path: Path, target: np.ndarray, synth: np.ndarray, result: dict, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr, block = SR, BLOCK
    pose = result["pose"].numpy()
    n_blocks = pose.shape[0]
    t = np.arange(n_blocks) * block / sr

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1, 1.4]})

    tt = np.arange(target.shape[0]) / sr
    axes[0].plot(tt, target, color="#2B7FA6", linewidth=0.5, label="input")
    axes[0].plot(tt, synth, color="#E28FA0", linewidth=0.5, alpha=0.85, label="resynthesised")
    for i, v in enumerate(result["voiced"]):
        if not v:
            axes[0].axvspan(i * block / sr, (i + 1) * block / sr, color="#DDDDDD", alpha=0.45)
    axes[0].set_title(f"{stem}: waveform (grey = unvoiced, original kept)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, result["f0"], color="#7A4FA6", linewidth=1.3)
    axes[1].scatter(t[result["voiced"]], result["f0"][result["voiced"]], s=6, color="#7A4FA6")
    axes[1].set_title("f0 contour")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)

    for name, colour in [("jaw", "#2B7FA6"), ("tbY", "#E28FA0"), ("lipAp", "#8CBF7A"), ("larynx", "#C9A227")]:
        axes[2].plot(t, pose[:, ARTIC_NAMES.index(name)], color=colour, linewidth=1.3, label=name)
    axes[2].set_title(f"tract trajectories (lambda={result['lam']})")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8, ncols=4)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"      plot: {path}", flush=True)


if __name__ == "__main__":
    main()
