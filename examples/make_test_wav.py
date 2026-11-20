"""Write a test wav from ctract itself: known articulators, known F0.

Ground truth for examples/fit_wav.py — the fit should recover this pose.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_wav import NEUTRAL, SR, glottal_flow, synthesize, write_wav  # noqa: E402

TRUE_POSE = list(NEUTRAL)
TRUE_POSE[0] = 0.25   # jaw more open
TRUE_POSE[1] = 3.35   # tongue body back
TRUE_POSE[2] = 6.10   # tongue body lower
TRUE_POSE[7] = 1.05   # lip aperture wider
TRUE_POSE[12] = 0.25  # larynx raised
F0 = 118.0

n = int(0.6 * SR)
pose = torch.tensor(TRUE_POSE, dtype=torch.float64)
flow = glottal_flow(F0, n, SR)
audio = synthesize(pose, flow, torch.tensor(1.0, dtype=torch.float64))

out = Path(__file__).resolve().parents[1] / "outputs/fit_wav"
out.mkdir(parents=True, exist_ok=True)
write_wav(out / "synthetic_vowel.wav", audio.numpy(), SR)
print("wrote", out / "synthetic_vowel.wav", "f0", F0)
print("pose:", {i: round(v, 3) for i, v in enumerate(TRUE_POSE)})
