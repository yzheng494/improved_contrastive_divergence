"""Standalone script to re-plot quantization comparison from saved quant_results.npy."""
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

# ── Config ──────────────────────────────────────────────────────────────────
npy_path   = "quant_results/quant_results.npy"
out_path   = "quant_results/quant_comparison.png"
fid_interval = 10   # must match what was used during inference
# ────────────────────────────────────────────────────────────────────────────

data = np.load(npy_path, allow_pickle=True).item()

quant_types = [k for k in data if k not in ('default_acc', 'default_fid', 'default_precision')]
default_acc       = data['default_acc']
default_fid       = data['default_fid']
default_precision = data['default_precision']

num_steps = len(default_acc)
steps     = list(range(num_steps))
fid_steps = [s for s in steps if s % fid_interval == 0 or s == num_steps - 1]

colors = {'fp4': 'tab:blue', 'fp8': 'tab:orange', 'int8': 'tab:green', 'nf4': 'tab:red'}

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# ── Panel 1: L2 divergence ───────────────────────────────────────────────────
ax = axes[0]
ax.axhline(0, color='black', linewidth=2.5, linestyle='--', label='default', zorder=10)
for qt in quant_types:
    ax.plot(steps, data[qt]['l2_div'], label=qt, color=colors.get(qt))
ax.set_xlabel('MCMC step')
ax.set_ylabel('Normalized L2 divergence')
ax.set_title('L2 divergence vs default trajectory')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 2: Inception Score ─────────────────────────────────────────────────
ax = axes[1]
for qt in quant_types:
    ax.plot(steps, data[qt]['accuracy'], label=qt, color=colors.get(qt), alpha=0.8)
ax.plot(steps, default_acc, label='default', color='black', linewidth=2.5, linestyle='--', zorder=10)
ax.set_xlabel('MCMC step')
ax.set_ylabel('Inception score (IS)')
ax.set_title('Inception score over MCMC steps (higher = more realistic & diverse)')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 3: FID ─────────────────────────────────────────────────────────────
ax = axes[2]
for qt in quant_types:
    ax.plot(fid_steps, [data[qt]['fid'][s] for s in fid_steps], label=qt, color=colors.get(qt), alpha=0.8)
ax.plot(fid_steps, [default_fid[s] for s in fid_steps], label='default', color='black', linewidth=2.5, linestyle='--', zorder=10)
ax.set_xlabel('MCMC step')
ax.set_ylabel('FID (lower = better)')
ax.set_title(f'FID vs real CelebA (every {fid_interval} steps)')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 4: Precision ───────────────────────────────────────────────────────
ax = axes[3]
for qt in quant_types:
    ax.plot(fid_steps, [data[qt]['precision'][s] for s in fid_steps], label=qt, color=colors.get(qt), alpha=0.8)
ax.plot(fid_steps, [default_precision[s] for s in fid_steps], label='default', color='black', linewidth=2.5, linestyle='--', zorder=10)
ax.set_xlabel('MCMC step')
ax.set_ylabel('Precision (higher = better)')
ax.set_title(f'Precision vs real CelebA (every {fid_interval} steps)')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
