"""Rebuild the resynthesis from a saved trajectory, block by block.

The whole-sentence render lets each block's excitation smear into its
neighbours, which is why the naive concatenation loses SNR.  Here every block
is rendered in isolation (its own three-block window) and the middle blocks are
cross-faded together, so the output really is the sum of block-local fits.

  python examples/resynth_blocks.py --json <trajectory.json> --wav <src.wav> --out <dir>
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import fit_sentence as F
from ctract_chimera import ARTIC_NAMES

torch.set_num_threads(8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ls-iters", type=int, default=25)
    args = ap.parse_args()

    rec = json.load(open(args.json))
    pose = torch.tensor([[b["articulators"][n] for n in ARTIC_NAMES] for b in rec["blocks"]])
    f0 = np.array([b["f0_hz"] for b in rec["blocks"]])
    n = pose.shape[0]
    sig, sr = F.read_wav(Path(args.wav))
    sig = F.resample(sig, sr, F.SR)[: n * F.BLOCK]

    solved = F.solve_flow_blocks(sig, pose, f0, iters=args.ls_iters)
    flow = solved["flow"]

    # block-wise render: each block alone in its three-block window
    out = np.zeros(n * F.BLOCK)
    fade = 128
    prev_tail = np.zeros(fade)
    for t in range(n):
        a = t * F.BLOCK
        seg_flow = torch.zeros(3 * F.BLOCK)
        seg_flow[F.BLOCK:2 * F.BLOCK] = flow[a:a + F.BLOCK]
        pose3 = pose[[max(0, t - 1), t, min(n - 1, t + 1)]]
        audio = F.render(pose3, seg_flow)[F.BLOCK:2 * F.BLOCK].numpy()
        out[a:a + F.BLOCK] = audio
        if t > 0:
            w = np.linspace(0, 1, fade)
            out[a:a + fade] = prev_tail * (1 - w) + audio[:fade] * w
        prev_tail = audio[-fade:].copy()

    snr = 10 * np.log10(float((sig ** 2).sum() / max(1e-12, float(((sig - out) ** 2).sum()))))
    metric = F.metrics(sig, out, np.array(rec["blocks"], dtype=object).astype(bool).astype(bool))
    print(f"block-wise resynthesis: snr {snr:.2f} dB  {metric}", flush=True)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    F.write_wav(outdir / "blocks_resynth.wav", out, F.SR)

    # parametric version for reference: the pulse train, per-block gain
    pulse_flow = F.glottal_flow(f0, n)
    audio_p = F.render(pose, pulse_flow).numpy()
    gains = (audio_p.reshape(n, F.BLOCK) * sig.reshape(n, F.BLOCK)).sum(1) \
        / (audio_p.reshape(n, F.BLOCK) ** 2).sum(1)
    audio_p = (audio_p.reshape(n, F.BLOCK) * gains[:, None]).reshape(-1)
    F.write_wav(outdir / "parametric_resynth.wav", audio_p, F.SR)
    print(f"parametric pulse: {F.metrics(sig, audio_p, np.array([b['voiced'] for b in rec['blocks']]))}",
          flush=True)


if __name__ == "__main__":
    main()
