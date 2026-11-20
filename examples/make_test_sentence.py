"""Synthetic sentence with a known articulator trajectory, for testing fit_sentence.py.

Three syllables (open /a/, spread high /i/, rounded /u/) with 120 ms smoothstep
transitions and a rising-then-falling f0 contour. The audio is rendered by the
same differentiable path the inversion uses, so any residual error is the
inversion's, not a model mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import fit_sentence as F

NEUTRAL_POSE = F.NEUTRAL


def smoothstep(x):
    return x * x * (3 - 2 * x)


def main() -> None:
    torch.set_num_threads(4)
    blocks_per_syllable, transition = 30, 8          # 30 * 16 ms = 480 ms, 8 * 16 = 128 ms
    syllables = [
        torch.tensor([0.30, 3.20, 6.05, 6.35, 7.20, 0.0, 0.0, 1.10, 0.15, 0.45, 0.0, -0.20, 0.20, 0.0, 0.0, 0.0, 0.0]),  # /a/
        torch.tensor([0.05, 4.90, 7.70, 6.20, 7.05, 0.0, 0.0, 1.70, 0.05, 0.95, 0.0, 0.20, -0.15, 0.0, 0.0, 0.0, 0.0]),  # /i/
        torch.tensor([0.35, 2.60, 6.40, 6.55, 7.45, 0.0, 0.0, 0.35, 0.35, 0.30, 0.0, -0.10, 0.35, 0.0, 0.0, 0.0, 0.0]),  # /u/
    ]
    f0_keys = [104.0, 148.0, 100.0]

    bounds = []
    start = 0
    for _ in syllables:
        bounds.append((start, start + blocks_per_syllable))
        start += blocks_per_syllable - transition
    n_blocks = bounds[-1][1]

    pose = torch.zeros(n_blocks, 17)
    f0 = np.zeros(n_blocks)
    for (a, b), target, key_f0 in zip(bounds, syllables, f0_keys):
        ramp = transition
        for i in range(a, b):
            pose[i] = target
            f0[i] = key_f0
        if a > 0:                                     # smoothstep blend from the previous syllable
            prev_end = a
            for j in range(ramp):
                w = smoothstep(torch.tensor((j + 1) / ramp))
                pose[prev_end - ramp + j] = torch.lerp(pose[prev_end - ramp - 1], target, w)
                f0[prev_end - ramp + j] = float(np.interp((j + 0.5) / ramp, [0, 1],
                                                          [f0[prev_end - ramp - 1], key_f0]))
    flow = F.glottal_flow(f0, n_blocks)
    with torch.no_grad():
        audio = F.render(pose, flow)
    out = Path(__file__).resolve().parents[1] / "outputs/fit_sentence"
    out.mkdir(parents=True, exist_ok=True)
    F.write_wav(out / "sentence_truth.wav", audio.numpy(), F.SR)
    gt = {"blocks": n_blocks, "block_ms": round(F.BLOCK / F.SR * 1000, 2),
          "f0": [round(float(v), 2) for v in f0],
          "poses": [[round(float(v), 4) for v in row] for row in pose]}
    (out / "sentence_truth.json").write_text(json.dumps(gt) + "\n")
    print(f"wrote {out / 'sentence_truth.wav'}  {n_blocks} blocks  {n_blocks * F.BLOCK / F.SR:.2f} s")


if __name__ == "__main__":
    main()
