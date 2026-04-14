#!/usr/bin/env python3
"""bars_sampling.py — Basin-Aware Rollback Sampling (BARS) for EBM composition.

Implements BARS and compares several configurations against fixed-step baselines.

BARS overview
─────────────
Phase 1 – EXPLORE: take large steps (η_hi) to descend fast.
Phase 2 – REFINE:  switch to small steps (η_lo = β·η_hi) after an overshoot is
           detected via a sliding-window oscillation ratio.  The sampler
           teleports back to x_best before entering REFINE.
Early termination: once REFINE makes no improvement for w consecutive steps,
           output x_best and stop (pad the trace with a flat tail).

Energy proxy used for BARS detection (and for the energy-vs-step plot):
    e_t = mean_batch( ‖∇_x Σ_i E_i(x_t)‖₂ )
This is the fixed-point residual; it converges to 0 at the data manifold.

Outputs (saved to --out_dir/)
──────────────────────────────
  bars_energy.png        — energy proxy vs sampling step (all configs)
  bars_is.png            — Inception Score vs sampling step (eval every N steps)
  bars_final_metrics.png — bar chart: final IS / FID / Precision
  bars_results.npy       — all per-step data dict

Usage
─────
    python bars_sampling.py
    python bars_sampling.py --num_steps 80 --num_samples 16 --out_dir bars_results
"""

import copy
import os
import os.path as osp
import sys
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from absl import flags
from PIL import Image
from torchvision import transforms

from models_2 import CelebAModel

# ─── Flags ────────────────────────────────────────────────────────────────────
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 4, 'Number of workers')
flags.DEFINE_string('logdir', 'cachedir', 'directory for logging')
flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_float('step_lr', 500.0, 'Langevin base step size (divided by num_models in main)')
flags.DEFINE_bool('cclass', True, 'conditional class')
flags.DEFINE_bool('proj_cclass', False, 'backwards compat')
flags.DEFINE_bool('spec_norm', True, 'spectral norm on weights')
flags.DEFINE_bool('use_bias', True, 'bias in conv')
flags.DEFINE_bool('use_attention', False, 'self attention in network')
flags.DEFINE_integer('num_steps', 60, 'Total MCMC sampling steps (BARS may terminate early)')
flags.DEFINE_string('task', 'bars', 'task type')
flags.DEFINE_bool('eval', False, 'quantitative eval')
flags.DEFINE_bool('latent_energy', False, 'latent energy in model')
flags.DEFINE_bool('proj_latent', False, 'projection of latents')
flags.DEFINE_bool('train', False, 'train mode')
flags.DEFINE_integer('num_samples', 64,
    'Images per trajectory batch.  Also used for IS evaluation at each step.')
flags.DEFINE_integer('mcmc_chunk', 16, 'Sub-batch size for chunked gradient computation')
flags.DEFINE_integer('eval_interval', 5, 'Compute IS/FID/Precision every this many steps')
flags.DEFINE_integer('fid_pca_dim', 64,
    'PCA dim before FID/Precision (0=off).  Strongly recommended when num_samples < 512 '
    'because the 2048-d Inception feature covariance is rank-deficient with few samples; '
    'PCA projects to a full-rank subspace.  Set to 0 only when num_samples >= 512.')
flags.DEFINE_string('out_dir', 'bars_results', 'Output directory')

FLAGS = flags.FLAGS

# ─── Model loading ────────────────────────────────────────────────────────────

def _load_model_list(models, resume_iters):
    model_list = []
    for model_name, resume_iter in zip(models, resume_iters):
        model_path = osp.join("cachedir", model_name, f"model_{resume_iter}.pth")
        checkpoint = torch.load(model_path)
        FLAGS_model = checkpoint['FLAGS']
        for k, v in dict(self_attn=False, multiscale=False,
                         alias=False, square_energy=False, cond=False).items():
            if not hasattr(FLAGS_model, k):
                FLAGS_model[k] = v
        m = CelebAModel(FLAGS_model)
        m.load_state_dict(checkpoint['ema_model_state_dict_0'])
        model_list.append(m.cuda().eval())
    return model_list


# ─── Gradient computation ─────────────────────────────────────────────────────

def _chunked_grad(im, model_list, labels, n):
    """Sum gradients across all models in sub-batches to bound peak GPU memory.

    Mathematically identical to a full-batch forward+backward because
    energy = sum over images (independent per-image terms).
    """
    chunk = FLAGS.mcmc_chunk
    grads = torch.zeros_like(im)
    for ci in range(0, n, chunk):
        x_c = im[ci:ci + chunk].detach().requires_grad_(True)
        lc = [l[ci:ci + chunk] for l in labels]
        e_c = sum(m.forward(x_c, l) for m, l in zip(model_list, lc))
        g_c = torch.autograd.grad([e_c.sum()], [x_c])[0]
        grads[ci:ci + chunk] = g_c.detach()
        del x_c, e_c, g_c
        torch.cuda.empty_cache()
    return grads  # detached, same shape as im


# ─── Inception / metric helpers ───────────────────────────────────────────────

_inception_feat_model = None
_inception_logit_model = None


def _get_feat_model():
    global _inception_feat_model
    if _inception_feat_model is None:
        from torchvision.models import inception_v3
        m = inception_v3(pretrained=True, transform_input=False)
        m.fc = nn.Identity()
        _inception_feat_model = m.cuda().eval()
    else:
        _inception_feat_model.cuda()
    return _inception_feat_model


def _extract_features(im_gpu, batch_size=16):
    """2048-d Inception pool features. im_gpu: (N,3,H,W) in [0,1]."""
    from torchvision import transforms as T
    model = _get_feat_model()
    resize = T.Resize((299, 299), antialias=True)
    feats = []
    with torch.no_grad():
        for i in range(0, im_gpu.shape[0], batch_size):
            out = model(resize(im_gpu[i:i + batch_size]))
            if hasattr(out, 'logits'):
                out = out.logits
            feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()


def _compute_is(im_gpu):
    """IS proxy: mean KL(p(y|x) || p(y)).  Higher = better."""
    import torch.nn.functional as F
    from torchvision import transforms as T
    from torchvision.models import inception_v3
    global _inception_logit_model
    if _inception_logit_model is None:
        _inception_logit_model = inception_v3(pretrained=True, transform_input=False).cuda().eval()
    else:
        _inception_logit_model.cuda()
    resize = T.Resize((299, 299), antialias=True)
    with torch.no_grad():
        logits = _inception_logit_model(resize(im_gpu))
        if hasattr(logits, 'logits'):
            logits = logits.logits
        p_yx = F.softmax(logits, dim=1)
        p_y  = p_yx.mean(0, keepdim=True)
        kl   = (p_yx * (p_yx.clamp(1e-8).log() - p_y.clamp(1e-8).log())).sum(1)
    return kl.mean().item()


def _compute_fid(feats_real, feats_fake):
    from scipy import linalg
    r, f = feats_real, feats_fake
    if FLAGS.fid_pca_dim > 0:
        from sklearn.decomposition import PCA
        nc = max(1, min(FLAGS.fid_pca_dim, r.shape[0] - 1, f.shape[0] - 1))
        pca = PCA(n_components=nc).fit(np.concatenate([r, f]))
        r, f = pca.transform(r), pca.transform(f)
    mu_r, sig_r = r.mean(0), np.cov(r, rowvar=False)
    mu_f, sig_f = f.mean(0), np.cov(f, rowvar=False)
    diff = mu_r - mu_f
    cov, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2 * cov))


def _compute_precision(feats_real, feats_fake, k=10):
    from sklearn.neighbors import NearestNeighbors
    r, f = feats_real, feats_fake
    if FLAGS.fid_pca_dim > 0:
        from sklearn.decomposition import PCA
        nc = max(1, min(FLAGS.fid_pca_dim, r.shape[0] - 1, f.shape[0] - 1))
        pca = PCA(n_components=nc).fit(np.concatenate([r, f]))
        r, f = pca.transform(r), pca.transform(f)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(r)
    radii = nbrs.kneighbors(r)[0][:, -1]
    dists, idx = nbrs.kneighbors(f, n_neighbors=k)
    return float((dists <= radii[idx]).any(axis=1).mean())


def _offload_inception():
    global _inception_feat_model, _inception_logit_model
    for m in [_inception_feat_model, _inception_logit_model]:
        if m is not None:
            m.cpu()
    torch.cuda.empty_cache()


# ─── Warm-up ──────────────────────────────────────────────────────────────────

def _warmup(model_list, labels, n, im_size):
    """10×20-step Langevin warm-up with augmentation restarts. Returns x0 on GPU."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    im = torch.rand(n, 3, im_size, im_size).cuda()

    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
    aug = transforms.Compose([
        transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
    im_noise = torch.randn_like(im)

    for _ in range(10):
        for _ in range(20):
            im_noise.normal_()
            im = (im + 0.001 * im_noise).detach()
            im_grad = _chunked_grad(im, model_list, labels, n)
            im = (im - FLAGS.step_lr * im_grad).clamp(0, 1)
        im_np  = (im.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        im_aug = [np.array(aug(Image.fromarray(im_np[i]))) for i in range(n)]
        im = torch.tensor(np.array(im_aug), dtype=torch.float32).cuda()

    return im.detach().clone()


# ─── Fixed-step trajectory ────────────────────────────────────────────────────

def _run_fixed(model_list, labels, x0, n, im_size, step_size):
    """Fixed step size MCMC.

    Returns
    -------
    trajectory : list[Tensor]  (CPU, one per step)
    energies   : list[float]   energy proxy e_t at each step (pre-update)
    """
    im = x0.clone()
    trajectory, energies = [], []

    for step in range(FLAGS.num_steps):
        g = torch.Generator(device='cuda')
        g.manual_seed(1000 + step)
        noise = torch.empty(n, 3, im_size, im_size, device='cuda').normal_(generator=g)
        im = (im + 0.001 * noise).detach()

        im_grad = _chunked_grad(im, model_list, labels, n)

        # Energy proxy: mean per-sample gradient L2-norm (pre-update)
        e_t = im_grad.norm(p=2, dim=(1, 2, 3)).mean().item()
        energies.append(e_t)

        im = (im - step_size * im_grad).clamp(0, 1).detach()
        trajectory.append(im.cpu())

    return trajectory, energies


# ─── BARS trajectory ──────────────────────────────────────────────────────────

def _run_bars(model_list, labels, x0, n, im_size,
              eta_hi, beta=0.1, window=10, psi_th=2.0):
    """BARS (Basin-Aware Rollback Sampling).

    Algorithm
    ─────────
    EXPLORE phase: use η_hi.  After *window* steps, check the oscillation ratio
      ψ = total_variation / |net_decrease|.  Trigger rollback when:
        • ψ ≥ ψ_th  (heavy zigzag oscillation), OR
        • net_decrease ≤ 0 AND e_t > e_best  (steady divergence past x*)

    On rollback: teleport x_t → x_best, switch to REFINE (η_lo = β·η_hi).
    REFINE phase: count *stale* steps (no improvement).  When stale ≥ window,
      declare done, teleport to x_best, pad trace with flat tail.

    Energy proxy: e_t = mean_batch( ‖∇_x ΣE_i(x_t)‖₂ )  (pre-update)

    Returns
    -------
    trajectory    : list[Tensor]  length == num_steps (padded after termination)
    energies      : list[float]   length == num_steps
    phases        : list[str]     'EXPLORE' | 'REFINE' | 'SWITCH' | 'DONE'
    switch_step   : int | None    step at which EXPLORE→REFINE triggered
    terminate_step: int | None    step at which early termination triggered
    """
    eps_f    = 1e-8
    eta_lo   = beta * eta_hi

    phase         = 'EXPLORE'
    x_best        = x0.detach().clone()   # GPU tensor
    e_best        = float('inf')
    e_buf         = deque(maxlen=window)  # sliding window of energies
    stale         = 0
    done          = False
    switch_step   = None
    terminate_step = None

    im = x0.clone()
    trajectory, energies, phases = [], [], []

    for step in range(FLAGS.num_steps):

        # ── Padding after early termination ───────────────────────────────────
        if done:
            trajectory.append(x_best.cpu())
            energies.append(e_best)
            phases.append('DONE')
            continue

        # ── Step 0: Langevin noise (deterministic per-step seed) ──────────────
        g = torch.Generator(device='cuda')
        g.manual_seed(1000 + step)
        noise = torch.empty(n, 3, im_size, im_size, device='cuda').normal_(generator=g)
        im = (im + 0.001 * noise).detach()

        # ── Compute gradient ───────────────────────────────────────────────────
        im_grad = _chunked_grad(im, model_list, labels, n)

        # ── Step 1: Energy proxy (pre-update) ─────────────────────────────────
        e_t = im_grad.norm(p=2, dim=(1, 2, 3)).mean().item()

        # ── Step 2: Update best-iterate cache ─────────────────────────────────
        if e_t < e_best:
            e_best = e_t
            x_best = im.detach().clone()
            stale  = 0
        elif phase == 'REFINE':
            stale += 1

        # ── Step 3: Slide window ───────────────────────────────────────────────
        e_buf.append(e_t)

        # ── Step 4: EXPLORE — overshoot detection (buffer must be full) ────────
        if phase == 'EXPLORE' and len(e_buf) == window:
            buf = list(e_buf)                       # oldest → newest
            D_t   = buf[0] - buf[-1]                # net decrease (+ = good)
            O_t   = sum(abs(buf[i] - buf[i-1]) for i in range(1, window))
            psi_t = O_t / (abs(D_t) + eps_f)

            trigger = (psi_t >= psi_th) or (D_t <= 0 and e_t > e_best)

            if trigger:
                phase       = 'REFINE'
                switch_step = step
                # Teleport: x_t → x_best
                im = x_best.clone()
                energies.append(e_t)
                phases.append('SWITCH')
                trajectory.append(im.cpu())
                continue

        # ── Step 5: REFINE — early termination ────────────────────────────────
        if phase == 'REFINE' and stale >= window:
            done           = True
            terminate_step = step
            im = x_best.clone()
            energies.append(e_t)
            phases.append('DONE')
            trajectory.append(im.cpu())
            continue

        # ── Step 6: Normal gradient step ──────────────────────────────────────
        eta = eta_lo if phase == 'REFINE' else eta_hi
        im  = (im - eta * im_grad).clamp(0, 1).detach()
        energies.append(e_t)
        phases.append(phase)
        trajectory.append(im.cpu())

    return trajectory, energies, phases, switch_step, terminate_step


# ─── Main evaluation ──────────────────────────────────────────────────────────

def bars_figure(model_list, select_idx):
    """Run all BARS and fixed configs, plot energy traces and final accuracies."""

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    n       = FLAGS.num_samples
    im_size = 128
    eta_base = FLAGS.step_lr   # already divided by num_models in __main__

    # ── Real image features (for FID / Precision) ─────────────────────────────
    import pandas as pd
    from torchvision import transforms as T

    # Attribute column/sign mapping (mirrors celeba_combine_2.py):
    # model order: [old, male, smiling, wavy_hair]
    # CelebA col indices: old=39, male=20, smiling=31, wavy_hair=33
    attr_cols  = [39, 20, 31, 33]
    attr_signs = [(-1, 1), (1, -1), (1, -1), (-1, 1)]  # (val when idx=1, val when idx=0)
    celeba_dir = "data/celeba/img_align_celeba"

    print("=== Loading real CelebA features for FID / Precision ===")
    attr_df = pd.read_csv("data/celeba/list_attr_celeba.txt", sep=r"\s+", skiprows=1)
    mask = np.ones(len(attr_df), dtype=bool)
    for col_idx, (v1, v0), six in zip(attr_cols, attr_signs, select_idx):
        col = attr_df.columns[col_idx]
        mask &= (attr_df[col].values == (v1 if six == 1 else v0))
    files = attr_df.index[mask].tolist()
    print(f"  {len(files)} attribute-matched images")
    np.random.seed(0)
    chosen = np.random.choice(files, size=min(2000, len(files)), replace=False)
    to_tensor = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    real_imgs = torch.stack(
        [to_tensor(Image.open(osp.join(celeba_dir, f)).convert('RGB')) for f in chosen]
    ).cuda()
    feats_real = _extract_features(real_imgs)
    print(f"  Real features: {feats_real.shape}\n")
    del real_imgs
    torch.cuda.empty_cache()

    # ── Labels ─────────────────────────────────────────────────────────────────
    labels = []
    for six in select_idx:
        lb = np.tile(np.eye(2)[six:six + 1], (n, 1))
        labels.append(torch.tensor(lb, dtype=torch.float32).cuda())

    # ── Warm-up → shared x0 ───────────────────────────────────────────────────
    _offload_inception()
    print("=== Warm-up → shared x0 ===")
    x0 = _warmup(model_list, labels, n, im_size)
    print("  x0 ready.\n")

    # ── Evaluate x0 once as the shared step-0 baseline ────────────────────────
    # traj[k] is the state AFTER k+1 gradient updates — so traj[0] already
    # differs across configs because they used different step sizes on the first
    # update.  Evaluating x0 itself and prepending it gives every config an
    # identical, meaningful origin in the metrics plot.
    print("=== Evaluating shared x0 (step-0 baseline) ===")
    _offload_inception()
    _x0_gpu   = x0.cuda()
    x0_is     = _compute_is(_x0_gpu)
    x0_feats  = _extract_features(_x0_gpu)
    x0_fid    = _compute_fid(feats_real, x0_feats)
    x0_prec   = _compute_precision(feats_real, x0_feats)
    del _x0_gpu, x0_feats
    torch.cuda.empty_cache()
    print(f"  x0: IS={x0_is:.3f}  FID={x0_fid:.1f}  Prec={x0_prec:.3f}\n")

    # ── Evaluation checkpoints ─────────────────────────────────────────────────
    # Start from eval_interval so traj index 0 (after 1 update) is not shown;
    # step 0 in the plot is always the shared x0 evaluated above.
    eval_steps = list(range(FLAGS.eval_interval, FLAGS.num_steps, FLAGS.eval_interval))
    if FLAGS.num_steps - 1 not in eval_steps:
        eval_steps.append(FLAGS.num_steps - 1)
    # x-axis positions used in all metric plots: 0 = x0, rest = traj checkpoints
    plot_steps = [0] + eval_steps

    # ── Configurations ─────────────────────────────────────────────────────────
    # BARS: (display_name, η_hi multiplier, β, window w, ψ_th)
    # β = η_lo / η_hi = 1x / 2x = 0.5  →  refine step = 1× base, explore step = 2× base
    # Sweep window w ∈ {2, 4, 6, 10, 15, 20} to study sensitivity.
    # One aggressive reference (16x → 1.6x) is kept for context.
    bars_configs = [
        ('BARS 2x→1x w=2',   2.0, 0.5,  2, 2.0),
        ('BARS 2x→1x w=4',   2.0, 0.5,  4, 2.0),
        ('BARS 2x→1x w=6',   2.0, 0.5,  6, 2.0),
        ('BARS 2x→1x w=10',  2.0, 0.5, 10, 2.0),
        ('BARS 2x→1x w=15',  2.0, 0.5, 15, 2.0),
        ('BARS 2x→1x w=20',  2.0, 0.5, 20, 2.0),
        ('BARS 16x β=0.1 w=10', 16.0, 0.1, 10, 2.0),
    ]

    # Fixed baselines: 1x (default), 2x, 4x
    fixed_mults = [1.0, 2.0, 4.0]

    results = {}   # name → metrics dict

    # ── Helper: evaluate a trajectory list ────────────────────────────────────
    def _eval_traj(traj):
        """Compute IS, FID, and Precision at every eval step (same as other experiments)."""
        is_vals, fid_vals, prec_vals = [], [], []
        for s in eval_steps:
            _offload_inception()
            im_gpu     = traj[s].cuda()
            is_v       = _compute_is(im_gpu)
            feats_fake = _extract_features(im_gpu)
            fid_v      = _compute_fid(feats_real, feats_fake)
            prec_v     = _compute_precision(feats_real, feats_fake)
            is_vals.append(is_v)
            fid_vals.append(fid_v)
            prec_vals.append(prec_v)
            del im_gpu
            torch.cuda.empty_cache()
        return is_vals, fid_vals, prec_vals

    # ── Run fixed baselines ────────────────────────────────────────────────────
    print("=== Fixed step-size baselines ===")
    for mult in fixed_mults:
        name = f'Fixed {int(mult)}x'
        step_size = eta_base * mult
        print(f"  {name} (η = {step_size:.1f}) ...", end=' ', flush=True)
        _offload_inception()

        m_copy = [copy.deepcopy(m) for m in model_list]
        traj, energies = _run_fixed(m_copy, labels, x0, n, im_size, step_size)
        del m_copy
        torch.cuda.empty_cache()

        is_vals, fid_vals, prec_vals = _eval_traj(traj)
        # Prepend shared x0 so step 0 is identical across all configs
        is_vals   = [x0_is]   + is_vals
        fid_vals  = [x0_fid]  + fid_vals
        prec_vals = [x0_prec] + prec_vals

        results[name] = {
            'energies':        energies,
            'is_at_eval':      is_vals,
            'fid_at_eval':     fid_vals,
            'prec_at_eval':    prec_vals,
            'final_is':        is_vals[-1],
            'final_fid':       fid_vals[-1],
            'final_precision': prec_vals[-1],
            'type':            'fixed',
            'mult':            mult,
        }
        print(f"IS={is_vals[-1]:.3f}  FID={fid_vals[-1]:.1f}  Prec={prec_vals[-1]:.3f}")

    print()

    # ── Run BARS configs ───────────────────────────────────────────────────────
    print("=== BARS configurations ===")
    for (name, mult, beta, window, psi_th) in bars_configs:
        eta_hi = eta_base * mult
        print(f"  {name} (η_hi={eta_hi:.1f}, β={beta}, w={window}) ...", end=' ', flush=True)
        _offload_inception()

        m_copy = [copy.deepcopy(m) for m in model_list]
        traj, energies, phases, sw_step, term_step = _run_bars(
            m_copy, labels, x0, n, im_size,
            eta_hi=eta_hi, beta=beta, window=window, psi_th=psi_th,
        )
        del m_copy
        torch.cuda.empty_cache()

        is_vals, fid_vals, prec_vals = _eval_traj(traj)
        # Prepend shared x0 so step 0 is identical across all configs
        is_vals   = [x0_is]   + is_vals
        fid_vals  = [x0_fid]  + fid_vals
        prec_vals = [x0_prec] + prec_vals

        sw_tag   = f"switch@{sw_step}"   if sw_step   is not None else "no switch"
        term_tag = f"term@{term_step}"   if term_step is not None else "no term"
        print(f"{sw_tag}, {term_tag} | IS={is_vals[-1]:.3f}  FID={fid_vals[-1]:.1f}  Prec={prec_vals[-1]:.3f}")

        results[name] = {
            'energies':        energies,
            'phases':          phases,
            'is_at_eval':      is_vals,
            'fid_at_eval':     fid_vals,
            'prec_at_eval':    prec_vals,
            'final_is':        is_vals[-1],
            'final_fid':       fid_vals[-1],
            'final_precision': prec_vals[-1],
            'type':            'bars',
            'mult':            mult,
            'beta':            beta,
            'window':          window,
            'switch_step':     sw_step,
            'terminate_step':  term_step,
        }

    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────────────────────────────────

    steps = list(range(FLAGS.num_steps))

    # Color palettes
    fixed_palette = {
        'Fixed 1x': '#1f77b4',   # steel blue  (dashed baseline)
        'Fixed 2x': '#2ca02c',   # green
        'Fixed 4x': '#d62728',   # red
    }
    # w-sweep: sequential purple→orange colormap for the 6 BARS 2x→1x configs
    import matplotlib.cm as _cm
    _w_colors = [_cm.plasma(v) for v in np.linspace(0.15, 0.85, 6)]
    bars_palette = {
        'BARS 2x→1x w=2':        _w_colors[0],
        'BARS 2x→1x w=4':        _w_colors[1],
        'BARS 2x→1x w=6':        _w_colors[2],
        'BARS 2x→1x w=10':       _w_colors[3],
        'BARS 2x→1x w=15':       _w_colors[4],
        'BARS 2x→1x w=20':       _w_colors[5],
        'BARS 16x β=0.1 w=10':  '#555555',   # dark grey — reference
    }

    bars_names  = [c[0] for c in bars_configs]
    fixed_names = [f'Fixed {int(m)}x' for m in fixed_mults]
    all_names   = fixed_names + bars_names

    # ── Figure 1: Energy proxy vs sampling step ────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    # Fixed baselines
    for name in fixed_names:
        r    = results[name]
        mult = r['mult']
        col  = fixed_palette[name]
        ls   = '--' if mult == 1.0 else '-'
        lw   = 2.5  if mult == 1.0 else 1.5
        ax.plot(steps, r['energies'], color=col, linestyle=ls, linewidth=lw,
                alpha=0.75, label=name)

    # BARS configs
    for name in bars_names:
        r      = results[name]
        col    = bars_palette[name]
        en     = r['energies']
        sw     = r['switch_step']
        term   = r['terminate_step']

        ax.plot(steps, en, color=col, linewidth=2.2, label=name)

        # Vertical dashed line at phase switch
        if sw is not None and sw < FLAGS.num_steps:
            ax.axvline(sw, color=col, linestyle=':', linewidth=1.0, alpha=0.7)
            y_ann = max(en[sw] * 1.01, en[sw] + 0.02 * (max(en) - min(en)))
            ax.annotate(
                f'switch@{sw}',
                xy=(sw, en[sw]), xytext=(sw + 0.5, y_ann),
                fontsize=7, color=col, va='bottom',
                arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
            )

        # Shaded region after termination (flat tail)
        if term is not None and term < FLAGS.num_steps:
            ax.axvspan(term, FLAGS.num_steps - 1, alpha=0.07, color=col)
            ax.annotate(
                f'term@{term}',
                xy=(term, en[term]), xytext=(term + 0.5, en[term] * 0.98),
                fontsize=7, color=col, va='top',
            )

    ax.set_xlabel('Sampling step', fontsize=12)
    ax.set_ylabel('Energy proxy  (mean ‖∇E‖₂)', fontsize=12)
    ax.set_title(
        'Energy proxy vs sampling step\n'
        'BARS (warm colors) vs Fixed step sizes (cool colors)\n'
        'Dotted vertical = EXPLORE→REFINE switch  |  Shaded = DONE (padded) region',
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    energy_path = osp.join(FLAGS.out_dir, 'bars_energy.png')
    plt.savefig(energy_path, dpi=150)
    plt.close()
    print(f"Saved {energy_path}")

    # ── Figure 2: IS / FID / Precision over sampling steps (3 panels) ───────────
    metric_over_steps = [
        ('is_at_eval',   'Inception Score (↑)',  False),
        ('fid_at_eval',  'FID (↓)',               True),
        ('prec_at_eval', 'Precision (↑)',          False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, ylabel, _invert) in zip(axes, metric_over_steps):
        for name in fixed_names:
            r    = results[name]
            mult = r['mult']
            col  = fixed_palette[name]
            ls   = '--' if mult == 1.0 else '-'
            lw   = 2.5  if mult == 1.0 else 1.5
            ax.plot(plot_steps, r[key], color=col, linestyle=ls, linewidth=lw,
                    alpha=0.8, label=name, marker='o', markersize=3)
        for name in bars_names:
            r   = results[name]
            col = bars_palette[name]
            ax.plot(plot_steps, r[key], color=col, linewidth=2.0,
                    label=name, marker='s', markersize=4)
        ax.set_xlabel('Sampling step', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} over sampling steps', fontsize=11)
        ax.legend(fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
        if key == 'prec_at_eval':
            ax.set_ylim(0, 1)

    fig.suptitle('BARS vs Fixed step sizes — metrics over sampling steps', fontsize=13)
    plt.tight_layout()
    is_path = osp.join(FLAGS.out_dir, 'bars_metrics_over_steps.png')
    plt.savefig(is_path, dpi=150)
    plt.close()
    print(f"Saved {is_path}")

    # ── Figure 3: Final task accuracy bar chart ────────────────────────────────
    final_is_vals   = [results[n]['final_is']        for n in all_names]
    final_fid_vals  = [results[n]['final_fid']       for n in all_names]
    final_prec_vals = [results[n]['final_precision'] for n in all_names]
    bar_colors      = [fixed_palette.get(n, bars_palette.get(n, 'gray')) for n in all_names]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    metric_cfgs = [
        (axes[0], final_is_vals,   'IS',        'Final Inception Score (↑)'),
        (axes[1], final_fid_vals,  'FID',       'Final FID (↓)'),
        (axes[2], final_prec_vals, 'Precision', 'Final Precision (↑)'),
    ]
    for ax, vals, ylabel, title in metric_cfgs:
        bars_plot = ax.bar(range(len(all_names)), vals, color=bar_colors,
                           edgecolor='black', linewidth=0.6)
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        # Value labels on bars
        for bar, val in zip(bars_plot, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) - min(vals)) * 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7,
            )

    fig.suptitle(
        f'Final task accuracies — BARS vs Fixed step sizes  ({FLAGS.num_steps} MCMC steps)',
        fontsize=12,
    )
    plt.tight_layout()
    metrics_path = osp.join(FLAGS.out_dir, 'bars_final_metrics.png')
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"Saved {metrics_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    col_w = max(len(n) for n in all_names) + 2
    print(f"\n{'─'*(col_w+55)}")
    print(f"{'Config':<{col_w}} | {'Final IS':>8} | {'Final FID':>9} | {'Final Prec':>10} | Notes")
    print(f"{'─'*(col_w+55)}")
    for name in all_names:
        r    = results[name]
        note = ''
        if r['type'] == 'bars':
            sw   = r.get('switch_step')
            term = r.get('terminate_step')
            note = (f"switch@{sw}" if sw   is not None else "no switch") + \
                   (f", term@{term}" if term is not None else "")
        print(f"{name:<{col_w}} | {r['final_is']:>8.4f} | {r['final_fid']:>9.2f}"
              f" | {r['final_precision']:>10.4f} | {note}")

    # ── Save all results ───────────────────────────────────────────────────────
    np.save(osp.join(FLAGS.out_dir, 'bars_results.npy'), results)
    print(f"\nAll results saved to {FLAGS.out_dir}/")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FLAGS(sys.argv)

    # Same model / attribute setup as celeba_combine_2.py
    models_orig       = ['celeba_128_male_2', 'celeba_128_old_2',
                         'celeba_128_smiling_2', 'celeba_128_wavy_hair_2']
    resume_iters_orig = ["latest", "6000", "13000", "9000"]

    # Composition: old ∧ male ∧ smiling ∧ ¬wavy_hair
    models       = [models_orig[1], models_orig[0], models_orig[2], models_orig[3]]
    resume_iters = [resume_iters_orig[1], resume_iters_orig[0],
                    resume_iters_orig[2], resume_iters_orig[3]]
    select_idx   = [1, 1, 1, 0]

    # Divide base step by number of composed models (matches existing scripts)
    FLAGS.step_lr = FLAGS.step_lr / len(models)

    model_list = _load_model_list(models, resume_iters)
    bars_figure(model_list, select_idx)
