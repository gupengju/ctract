"""Plot the L-BFGS convergence and recovered-vs-true controls for fit_synthetic."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_synthetic import fit  # noqa: E402

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
result = fit()

row = json.loads((root / "tests/fixtures/fit.json").read_text())
true_artic = row["artic"]
names = ["jaw", "lip_aperture", "larynx", "flow_gain"]
true_vals = [true_artic[0], true_artic[7], true_artic[12], 1.0]
fit_vals = [
    result.parameters["jaw"],
    result.parameters["lip_aperture"],
    result.parameters["larynx"],
    result.parameters["flow_gain"],
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.semilogy(result.history, color="#2B7FA6", linewidth=1.6)
ax1.set_xlabel("L-BFGS iteration")
ax1.set_ylabel("Spectral loss (log)")
ax1.set_title("Convergence: %.4g -> %.3g" % (result.initial_loss, result.final_loss))
ax1.grid(alpha=0.3)

x = range(len(names))
w = 0.38
ax2.bar([i - w / 2 for i in x], true_vals, width=w, label="true", color="#8BC8EA")
ax2.bar([i + w / 2 for i in x], fit_vals, width=w, label="recovered", color="#C9A7E8")
ax2.set_xticks(list(x))
ax2.set_xticklabels(names, rotation=15)
ax2.set_title("Recovered controls vs ground truth")
ax2.legend()
ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = root / "outputs/fit_result.png"
fig.savefig(out, dpi=150)
print("saved", out)
print(json.dumps({"true": dict(zip(names, true_vals)),
                  "recovered": result.parameters}, indent=2))
