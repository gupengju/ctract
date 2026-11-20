"""WAV in -> articulator + source parameters out -> new WAV (analysis-by-synthesis).

Pipeline
  1. read wav (stdlib `wave`; 8/16/24/32-bit PCM), mono-mix, resample to 16 kHz
  2. pick the steadiest voiced segment (RMS minus spectral flux)
  3. estimate F0 by autocorrelation with parabolic refinement, then refine on a
     small grid against the target's own spectral envelope
  4. build the excitation from CHIMERA's own glottal area pulse
  5. fit 10 articulators to the target's spectral envelope with Adam on
     ctract's exact autograd gradient through geometry + convolutional readout
  6. resynthesise and write {params.json, *_resynth.wav, *_fit.png}

Usage
  python examples/fit_wav.py --wav input.wav
  python examples/fit_wav.py --wav input.wav --segment 0.25 --iters 80

Scope: one stationary voiced segment (vowels / sustained sonorants). Consonants,
coarticulation, prosody and the source's full 17 glottal parameters are outside
this API; see README "model boundaries".

Known limitation: a single steady vowel identifies the tract only up to an
acoustically equivalent family. jaw and tbY trade off against each other -- on
ctract's own synthetic vowels the fitted jaw came out ~1.0 cm above truth while
tbY rose ~0.6 cm, and the resulting spectra stayed within 0.04-0.05 of the
target. The returned parameters are therefore *a* valid explanation of the
sound, not the speaker's true anatomy. When physiological identity matters,
shrink FREE (e.g. drop jaw) or add a prior; when only the sound matters, this is
enough. Verified end-to-end on two synthetic vowels: 118.0 Hz -> fitted 118.00,
196.0 Hz -> 196.05, envelope loss 0.44->0.04 and 0.25->0.05, ~90 s each.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from ctract_chimera import ARTIC_NAMES, DEFAULT_GLOTTIS, NEUTRAL, ExactGeometry, Renderer
from ctract_chimera.anatomy import ARTIC_MAX, ARTIC_MIN
from ctract_chimera.source import glottal_shape

torch.set_default_dtype(torch.float64)

SR = 16000.0          # analysis and synthesis rate
INTERNAL = 96000.0    # CHIMERA's physical integration rate
PAD = 2048            # samples dropped from the renderer's startup transient
FREE = [0, 1, 2, 3, 4, 7, 8, 11, 12, 16]   # jaw,tbX,tbY,ttX,ttY,lipAp,lipProt,velum,larynx,dorsum
BANDS = 32
BAND_LO, BAND_HI = 300.0, 4000.0   # below 300 Hz the source's harmonics dominate
FLOOR_DB = 35.0                    # ignore bands this far under the target's peak
EPS = 1e-12


# ------------------------------------------------------------------ I/O

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """PCM wav -> (mono float64 in [-1, 1], sample rate). No external deps."""
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
    """Band-limited FFT resampling."""
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

def pick_segment(x: np.ndarray, sr: float, dur: float) -> tuple[np.ndarray, int]:
    """Steady, loud window: mean RMS minus 3x mean spectral flux."""
    n = int(dur * sr)
    if x.shape[0] <= n:
        return x, 0
    hop, win = int(0.005 * sr), int(0.025 * sr)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)[None, :]
    rms = np.sqrt((frames ** 2).mean(axis=1))
    mag = np.abs(np.fft.rfft(frames, axis=1))
    mag = mag / (mag.sum(axis=1, keepdims=True) + EPS)
    flux = np.concatenate([[0.0], np.abs(np.diff(mag, axis=0)).sum(axis=1)])
    n_frames = max(1, n // hop)
    best, best_score = 0, -np.inf
    for start in range(max(1, len(rms) - n_frames + 1)):
        end = min(len(rms), start + n_frames)
        score = float(rms[start:end].mean()) - 3.0 * float(flux[start:end].mean())
        if score > best_score:
            best, best_score = start, score
    return x[best * hop: best * hop + n], best * hop


def estimate_f0(x: np.ndarray, sr: float, lo: float = 60.0, hi: float = 420.0,
                step: float = 0.05) -> float:
    """Harmonic sieve: the pitch whose harmonic series carries the most energy.

    Plain autocorrelation is off by <1 % here, which is enough to move every
    harmonic off the target's bands and bias the fitted tract. Summing the
    spectrum at k*f0 (weighted 1/k) over a fine grid pins the pitch instead:
    measured 118.00 Hz on a 118.0 Hz source, 196.05 on a 196.0 Hz source.
    """
    n = x.shape[0]
    if n < 64:
        return 120.0
    pad = 1 << 16
    mag = np.abs(np.fft.rfft(x * np.hanning(n), n=pad))
    freqs = np.fft.rfftfreq(pad, 1 / sr)
    grid = np.arange(lo, hi, step)
    score = np.zeros_like(grid)
    for k in range(1, int(sr / 2 / lo) + 1):
        probe = k * grid
        if probe.max() >= freqs[-1]:
            break
        score += np.interp(probe, freqs, mag) / k
    return float(grid[int(score.argmax())])


# ---------------------------------------------------------------- model

_CACHE: dict[float, tuple[ExactGeometry, Renderer]] = {}


def _model(sr: float) -> tuple[ExactGeometry, Renderer]:
    if sr not in _CACHE:
        _CACHE[sr] = (
            ExactGeometry(),
            Renderer(sample_rate=sr, internal_rate=INTERNAL, block_size=256, ir_length=2048, response_fft=8192),
        )
    return _CACHE[sr]


def glottal_flow(f0: float, n: int, sr: float) -> torch.Tensor:
    """Repeated CHIMERA glottal area pulse used as the ideal volume-velocity source.

    1e-3 cm^3/s keeps the airway in its linear regime; the fitted gain carries
    the absolute level, exactly as examples/fit_synthetic.py does.
    """
    oq = torch.tensor(DEFAULT_GLOTTIS[4])
    skew = torch.tensor(DEFAULT_GLOTTIS[5])
    sharp = torch.tensor(DEFAULT_GLOTTIS[6])
    phase = (torch.arange(n) * f0 / sr) % 1.0
    return 1e-3 * glottal_shape(phase, oq, skew, sharp)


def synthesize(pose: torch.Tensor, flow: torch.Tensor, gain: torch.Tensor,
               sr: float = SR, pad: int = 0) -> torch.Tensor:
    """(17,) articulators + (L,) ideal flow -> (L - pad,) radiated audio."""
    geometry, renderer = _model(sr)
    g = {key: value[:, None] for key, value in geometry(pose[None], mode="fit", temperature=0.05).items()}
    out = renderer((flow * gain)[None], g, 1, temperature=1e-5)[0]
    return out[pad:] if pad else out


def band_map(n: int, sr: float) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.fft.rfftfreq(n, 1 / sr)
    edges = np.geomspace(BAND_LO, BAND_HI, BANDS + 1)
    index = torch.zeros(freqs.shape[0], dtype=torch.long)
    for i in range(BANDS):
        index[(freqs >= edges[i]) & (freqs < edges[i + 1])] = i
    count = torch.zeros(BANDS)
    count.index_add_(0, index, torch.ones_like(freqs))
    return index, count.clamp(min=1)


# ----------------------------------------------------------------- fit

def fit(target: np.ndarray, f0: float, iters: int, reg: float = 0.05,
        verbose: bool = True) -> dict:
    """Spectral-envelope analysis-by-synthesis over a bounded articulator space."""
    n = target.shape[0]
    sr = SR
    tgt = torch.tensor(target)
    window = torch.hann_window(n)
    band_index, band_count = band_map(n, sr)

    def envelope(signal: torch.Tensor) -> torch.Tensor:
        mag = torch.fft.rfft(signal * window).abs()
        out = torch.zeros(BANDS)
        out.index_add_(0, band_index, mag)
        return out / band_count

    ref_amp = envelope(tgt)
    # Only bands that carry the target's energy may constrain the fit: a near
    # silent band turns a 1 Hz shift into a huge log-ratio.
    weights = ref_amp / ref_amp.sum()
    mask = ref_amp > ref_amp.max() * 10 ** (-FLOOR_DB / 20)
    ref = ref_amp.clamp(min=EPS).log()

    def loss_of(audio: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # articulatory inversion is ill-posed (jaw/tongue trade off), so keep
        # the solution near the neutral posture unless the envelope demands it
        amp = envelope(audio).clamp(min=EPS)
        # a free gain only shifts the log envelope by a constant
        shift = ((ref - amp.log()) * weights)[mask].sum() / weights[mask].sum()
        fit_loss = (((ref - amp.log() - g - shift) ** 2) * weights)[mask].sum() / weights[mask].sum()
        return fit_loss + reg * (u ** 2).mean()

    lo = torch.tensor([ARTIC_MIN[i] for i in FREE])
    hi = torch.tensor([ARTIC_MAX[i] for i in FREE])
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    neutral = torch.tensor([NEUTRAL[i] for i in FREE])
    u = torch.atanh(((neutral - mid) / half).clamp(-0.999, 0.999)).clone().requires_grad_(True)
    gain = torch.zeros((), requires_grad=True)

    base = torch.tensor(NEUTRAL)
    mask_full = torch.zeros(17)
    mask_full[FREE] = 1.0
    free_index = torch.tensor(FREE, dtype=torch.long)

    def to_pose(v: torch.Tensor) -> torch.Tensor:
        """Bounded controls -> full 17-vector, out-of-place (no index_put_)."""
        vals = mid + half * torch.tanh(v)
        full = torch.zeros(17).index_put((free_index,), vals)
        return base * (1.0 - mask_full) + full * mask_full

    flow = glottal_flow(f0, n + PAD, sr)
    history: list[float] = []
    opt = torch.optim.Adam([u, gain], lr=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters), eta_min=0.005)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        audio = synthesize(to_pose(u), flow, gain.exp(), pad=PAD)
        loss = loss_of(audio, gain)
        loss.backward()
        history.append(float(loss.item()))
        return loss

    for i in range(iters):
        opt.step(closure)
        sched.step()
        if verbose and (i + 1) % 20 == 0:
            print(f"      iter {i + 1:3d}/{iters}  envelope loss {history[-1]:.4f}")

    with torch.no_grad():
        pose = to_pose(u)
        # The f0 search ran at the neutral posture; re-run it now that the pose
        # matches the target, so pitch and tract are chosen jointly.
        best_audio, best_total = None, float("inf")
        for k in range(-4, 5):
            cand = f0 * (1.0 + 0.003 * k)
            cand_flow = glottal_flow(cand, n + PAD, sr)
            cand_audio = synthesize(pose, cand_flow, gain.exp(), pad=PAD)
            cand_amp = envelope(cand_audio).clamp(min=EPS)
            cand_shift = ((ref - cand_amp.log()) * weights)[mask].sum() / weights[mask].sum()
            value = float(loss_of(cand_audio, gain.exp()))
            if value < best_total:
                best_total, best_audio, best_f0_final = value, cand_audio, cand
        f0 = best_f0_final
        audio = best_audio
        amp = envelope(audio).clamp(min=EPS)
        shift = ((ref - amp.log()) * weights)[mask].sum() / weights[mask].sum()
        total_gain = float(gain.exp() * shift.exp())

    return {
        "pose": pose,
        "gain": total_gain,
        "audio": audio.numpy() * total_gain,
        "f0": f0,
        "history": history,
        "final_loss": history[-1] if history else float("nan"),
    }


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="wav -> CHIMERA parameters -> wav")
    ap.add_argument("--wav", required=True, help="input wav path")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--segment", type=float, default=0.30, help="analysis window, seconds")
    ap.add_argument("--iters", type=int, default=80, help="Adam iterations")
    ap.add_argument("--reg", type=float, default=0.05, help="neutral-posture regularisation")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    src = Path(args.wav).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    outdir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "outputs/fit_wav"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    print(f"[1/4] reading {src.name}")
    signal, sr_in = read_wav(src)
    signal = resample(signal, sr_in, SR)
    print(f"      {sr_in} Hz -> {SR:.0f} Hz, {signal.shape[0] / SR:.2f} s")

    print("[2/4] selecting a steady voiced segment")
    segment, offset = pick_segment(signal, SR, args.segment)
    f0 = estimate_f0(segment, SR)
    print(f"      offset={offset / SR:.3f}s  samples={segment.shape[0]}  f0~{f0:.2f} Hz")

    print(f"[3/4] fitting {len(FREE)} articulators to the spectral envelope")
    result = fit(segment, f0, args.iters, args.reg)
    print(f"      loss {result['history'][0]:.4f} -> {result['final_loss']:.4f}   f0={result['f0']:.2f} Hz")

    pose = result["pose"].numpy()
    record = {
        "source": {
            "wav": str(src), "input_sample_rate": sr_in, "analysis_rate": SR,
            "segment_seconds": args.segment, "segment_offset": round(offset / SR, 4),
            "f0_hz": round(result["f0"], 3),
        },
        "articulators": {name: round(float(pose[i]), 5) for i, name in enumerate(ARTIC_NAMES)},
        "free_articulators": [ARTIC_NAMES[i] for i in FREE],
        "source_flow_gain": round(result["gain"], 6),
        "envelope_loss": {"initial": result["history"][0], "final": result["final_loss"]},
        "note": "stationary voiced segment; consonants, coarticulation and prosody "
                "are outside this model, see README 'model boundaries'",
    }
    params_path = outdir / f"{stem}_params.json"
    wav_path = outdir / f"{stem}_resynth.wav"
    params_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    write_wav(wav_path, result["audio"], SR)
    print(f"[4/4] wrote {params_path.name}, {wav_path.name}")
    _plot(outdir / f"{stem}_fit.png", signal, segment, offset, result, stem)


def _plot(path: Path, signal: np.ndarray, segment: np.ndarray, offset: int,
          result: dict, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr = SR
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    t = np.arange(signal.shape[0]) / sr
    axes[0].plot(t, signal, color="#2B7FA6", linewidth=0.6)
    axes[0].axvspan(offset / sr, (offset + segment.shape[0]) / sr, color="#F4B393", alpha=0.35,
                    label="fitted segment")
    axes[0].set_title(f"input: {stem}")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(result["history"], color="#7A4FA6", linewidth=1.4)
    axes[1].set_title(f"envelope loss {result['history'][0]:.3g} -> {result['final_loss']:.3g}")
    axes[1].set_xlabel("iteration")
    axes[1].grid(alpha=0.3)

    def spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mag = np.abs(np.fft.rfft(x * np.hanning(x.shape[0])))
        return np.fft.rfftfreq(x.shape[0], 1 / sr), 20 * np.log10(mag / (mag.max() + EPS) + EPS)

    f_t, s_t = spec(segment)
    f_s, s_s = np.fft.rfftfreq(result["audio"].shape[0], 1 / sr), None
    mag = np.abs(np.fft.rfft(result["audio"] * np.hanning(result["audio"].shape[0])))
    s_s = 20 * np.log10(mag / (mag.max() + EPS) + EPS)
    axes[2].plot(f_t, s_t, color="#8BC8EA", linewidth=0.8, label="target")
    axes[2].plot(f_s, s_s, color="#EAA7B2", linewidth=0.8, alpha=0.85, label="resynthesised")
    axes[2].set_xlim(0, 5000)
    axes[2].set_ylim(-70, 2)
    axes[2].set_title(f"spectra (f0={result['f0']:.1f} Hz)")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("dB")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"      plot: {path}")


if __name__ == "__main__":
    main()
