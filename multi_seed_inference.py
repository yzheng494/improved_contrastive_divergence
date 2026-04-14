"""Multi-seed inference for EBM composition with optional quantization.

Evaluates IS, FID, Precision at every --eval_interval MCMC steps.
At each checkpoint all N seeds are scored; the best one is compared against
a single-seed baseline.  When --quant_type is set the models are quantized
first, and an unquantized single-seed reference line is also shown.

Each metric (IS, FID, Precision) is saved as its own figure so plots stay
readable.  A 4th figure shows the per-seed energy heatmap.

Usage:
    # Plain multi-seed
    python multi_seed_inference.py --num_seeds 8 --num_steps 200 --eval_interval 50

    # With quantization (adds unquantized reference line)
    python multi_seed_inference.py --num_seeds 8 --quant_type nf4
    python multi_seed_inference.py --num_seeds 8 --quant_type fp4
    python multi_seed_inference.py --num_seeds 8 --quant_type int8
    python multi_seed_inference.py --num_seeds 8 --quant_type fp8
"""

import copy
import math
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from absl import flags
from PIL import Image
from scipy.misc import imsave
from torchvision import transforms

from models_2 import CelebAModel

# ── Flags ──────────────────────────────────────────────────────────────────────
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 4, 'Number of workers')
flags.DEFINE_string('logdir', 'cachedir', 'directory for logging')
flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_float('step_lr', 500.0, 'Langevin step size')
flags.DEFINE_bool('cclass', True, 'conditional class')
flags.DEFINE_bool('proj_cclass', False, 'backwards compat')
flags.DEFINE_bool('spec_norm', True, 'spectral norm on weights')
flags.DEFINE_bool('use_bias', True, 'bias in conv')
flags.DEFINE_bool('use_attention', False, 'self attention in network')
flags.DEFINE_integer('num_steps', 200, 'Total MCMC refinement steps')
flags.DEFINE_string('task', 'combination_figure', 'task type')
flags.DEFINE_bool('eval', False, 'quantitative eval')
flags.DEFINE_bool('latent_energy', False, 'latent energy in model')
flags.DEFINE_bool('proj_latent', False, 'projection of latents')
flags.DEFINE_bool('train', False, 'train mode')

flags.DEFINE_integer('num_seeds', 8,
    'Number of independent MCMC chains to run (8 or 16)')
flags.DEFINE_integer('num_samples', 16,
    'Images per chain (batch size)')
flags.DEFINE_integer('num_trials', 5,
    'Evaluation trials to average over')
flags.DEFINE_integer('eval_interval', 50,
    'Evaluate IS/FID/Precision every this many MCMC steps')
flags.DEFINE_integer('fid_pca_dim', 0,
    'PCA dim before FID/Precision (0=off; 64 recommended when num_samples<512)')
flags.DEFINE_string('out_dir', 'multi_seed_results',
    'Directory for output images and plots')
flags.DEFINE_string('score_metric', 'is',
    'Criterion to pick the best seed: "is" | "energy" | "fid" | "precision"')
flags.DEFINE_string('quant_type', 'none',
    'Weight quantization: "none" | "fp4" | "fp8" | "int8" | "nf4"')
flags.DEFINE_integer('mcmc_chunk_size', 64,
    'Sub-batch size for chunked MCMC gradient computation (reduces peak GPU memory)')
flags.DEFINE_bool('rank_comparison', True,
    'When num_seeds>=4: compare worst/rank-N//4/rank-N//2/best seeds as proxies '
    'for 1-seed/4-seed/8-seed/N-seed experiments instead of single-vs-multi.')

FLAGS = flags.FLAGS


# ── Quantization (from celeba_combine_2.py) ────────────────────────────────────

def quantize_model(model, quant_type):
    """Replace nn.Linear layers with quantized versions in-place."""
    import bitsandbytes as bnb
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            has_bias = module.bias is not None
            if quant_type in ('nf4', 'fp4'):
                ql = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias, quant_type=quant_type)
                ql.weight = bnb.nn.Params4bit(
                    module.weight.data.clone(), requires_grad=False, quant_type=quant_type)
                if has_bias:
                    ql.bias = nn.Parameter(module.bias.data.clone())
                ql = ql.cuda()
                setattr(model, name, ql)
            elif quant_type == 'int8':
                w = module.weight.data.float()
                scale = 127.0 / w.abs().max().clamp(min=1e-12)
                w_int8 = (w * scale).round().clamp(-128, 127)
                module.weight.data = (w_int8 / scale).to(module.weight.data.dtype)
            elif quant_type == 'fp8':
                w = module.weight.data.float()
                fp8_max = 448.0
                scale = fp8_max / w.abs().max().clamp(min=1e-12)
                w_q = ((w * scale).clamp(-fp8_max, fp8_max) * 8).round() / 8
                module.weight.data = (w_q / scale).to(module.weight.data.dtype)
            else:
                raise ValueError(f"Unknown quant_type: {quant_type}")
        else:
            quantize_model(module, quant_type)
    return model


# ── Inception helpers (from celeba_combine_2.py) ───────────────────────────────

_inception_feat_model   = None
_inception_logit_model  = None


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


def _extract_features(images_cuda, batch_size=16):
    """2048-d Inception pool features. images_cuda: (N,3,H,W) in [0,1]."""
    from torchvision import transforms as T
    model  = _get_feat_model()
    resize = T.Resize((299, 299), antialias=True)
    feats  = []
    with torch.no_grad():
        for i in range(0, images_cuda.shape[0], batch_size):
            out = model(resize(images_cuda[i:i+batch_size]))
            if hasattr(out, 'logits'):
                out = out.logits
            feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()


def _maybe_pca(r, f):
    d = FLAGS.fid_pca_dim
    if d <= 0:
        return r, f
    from sklearn.decomposition import PCA
    nc = max(1, min(d, r.shape[0]-1, f.shape[0]-1, r.shape[1]))
    pca = PCA(n_components=nc).fit(np.concatenate([r, f]))
    return pca.transform(r), pca.transform(f)


def _compute_fid(feats_real, feats_fake):
    from scipy import linalg
    r, f = _maybe_pca(feats_real, feats_fake)
    mu_r, sig_r = r.mean(0), np.cov(r, rowvar=False)
    mu_f, sig_f = f.mean(0), np.cov(f, rowvar=False)
    diff = mu_r - mu_f
    cov, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2 * cov))


def _compute_precision(feats_real, feats_fake, k=10):
    from sklearn.neighbors import NearestNeighbors
    r, f = _maybe_pca(feats_real, feats_fake)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(r)
    radii = nbrs.kneighbors(r)[0][:, -1]
    nd    = nbrs.kneighbors(f, n_neighbors=1, return_distance=True)[0][:, 0]
    ni    = nbrs.kneighbors(f, n_neighbors=1, return_distance=False)[:, 0]
    return float((nd <= radii[ni]).mean())


def _compute_is(im_gpu):
    """IS proxy: mean KL(p(y|x) || p(y)). Higher = better."""
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


def _offload_inception():
    global _inception_feat_model, _inception_logit_model
    for m in [_inception_feat_model, _inception_logit_model]:
        if m is not None:
            m.cpu()
    torch.cuda.empty_cache()


def _score_batch(im_cpu, feats_real):
    """Compute IS, FID, Precision for one batch. Returns dict."""
    im_gpu  = im_cpu.cuda()
    is_val  = _compute_is(im_gpu)
    feats_f = _extract_features(im_gpu)
    fid_val = _compute_fid(feats_real, feats_f)
    pre_val = _compute_precision(feats_real, feats_f)
    im_gpu.cpu()
    torch.cuda.empty_cache()
    return {'is': is_val, 'fid': fid_val, 'precision': pre_val}


# ── Real image features ────────────────────────────────────────────────────────

def _load_real_features(select_idx):
    import pandas as pd
    from torchvision import transforms as T

    # Column/sign mapping mirrors celeba_combine_2.py quantization_figure
    attr_cols  = [39, 20, 31, 33]
    attr_signs = [(-1, 1), (1, -1), (1, -1), (-1, 1)]

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
    imgs = torch.stack([to_tensor(Image.open(
        osp.join("data/celeba/img_align_celeba", f)).convert('RGB')) for f in chosen]).cuda()
    feats = _extract_features(imgs)
    print(f"  Real features: {feats.shape}\n")
    del imgs
    torch.cuda.empty_cache()
    return feats


# ── MCMC helpers ───────────────────────────────────────────────────────────────

def _get_color_distortion(s=1.0):
    cj  = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
    rcj = transforms.RandomApply([cj], p=0.8)
    rg  = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rcj, rg])


def _mcmc_step(im, model_list, labels, n):
    """One Langevin gradient step, chunked to bound peak GPU memory.

    Processes images in sub-batches of mcmc_chunk_size; since energy = sum over
    images the per-image gradients are independent, so this is mathematically
    identical to a full-batch step.

    Returns (updated_im_detached, mean_energy_float).
    """
    chunk  = FLAGS.mcmc_chunk_size
    grads  = torch.zeros_like(im)
    total_e = 0.0

    for ci in range(0, n, chunk):
        x_c = im[ci:ci+chunk].detach().requires_grad_(True)
        lc  = [l[ci:ci+chunk] if l is not None else None for l in labels]
        e_c = sum(m.forward(x_c, l) for m, l in zip(model_list, lc))
        g_c = torch.autograd.grad([e_c.sum()], [x_c])[0]
        grads[ci:ci+chunk] = g_c.detach()
        total_e += e_c.detach().sum().item()
        del x_c, e_c, g_c
        torch.cuda.empty_cache()

    im_new = (im - FLAGS.step_lr * grads).clamp(0, 1).detach()
    return im_new, total_e / n


def _warmup(model_list, labels, n, im_size, seed):
    """10×20-step warm-up with augmentation restarts. Returns x0 on GPU."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    im      = torch.rand(n, 3, im_size, im_size).cuda()
    im_noise = torch.randn_like(im).detach()
    aug = transforms.Compose([
        transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        _get_color_distortion(),
        transforms.ToTensor(),
    ])

    for _ in range(10):
        for _ in range(20):
            im_noise.normal_()
            im = (im + 0.001 * im_noise).detach()
            im, _ = _mcmc_step(im, model_list, labels, n)
        im_np  = (im.detach().cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)
        im_aug = [np.array(aug(Image.fromarray(im_np[i]))) for i in range(n)]
        im = torch.tensor(np.array(im_aug), dtype=torch.float32).cuda()

    return im.detach()


def _refine_checkpoints(model_list, labels, x0, n, im_size, seed, eval_steps):
    """Langevin refinement; saves a CPU copy of images at each step in eval_steps.

    eval_steps: set/list of 1-indexed step numbers to snapshot (e.g. {50,100,200}).
    Returns ({step: im_cpu}, final_energy).
    """
    torch.manual_seed(seed + 10000)
    torch.cuda.manual_seed(seed + 10000)

    im       = x0.clone()
    im_noise = torch.randn_like(im).detach()
    eval_set = set(eval_steps)
    ckpts    = {}
    final_energy = None

    for step in range(1, FLAGS.num_steps + 1):
        im_noise.normal_()
        im = (im + 0.001 * im_noise).detach()
        im, mean_e = _mcmc_step(im, model_list, labels, im.shape[0])

        if step in eval_set:
            ckpts[step] = im.detach().cpu()
        if step == FLAGS.num_steps:
            final_energy = mean_e

    return ckpts, final_energy


# ── Main function ─────────────────────────────────────────────────────────────

def multi_seed_inference(model_list_orig, select_idx):
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    n          = FLAGS.num_samples
    im_size    = 128
    num_seeds  = FLAGS.num_seeds
    num_trials = FLAGS.num_trials
    quant_type = FLAGS.quant_type

    # Eval checkpoints: every eval_interval steps, always including final step
    eval_steps = list(range(FLAGS.eval_interval, FLAGS.num_steps, FLAGS.eval_interval))
    if FLAGS.num_steps not in eval_steps:
        eval_steps.append(FLAGS.num_steps)
    eval_steps = sorted(eval_steps)
    print(f"Eval checkpoints: {eval_steps}")

    # Build model lists
    # model_list_exp: the models actually used for all seeds
    # model_list_ref: unquantized reference (only used when quant_type != 'none')
    if quant_type != 'none':
        model_list_exp = [quantize_model(copy.deepcopy(m), quant_type)
                          for m in model_list_orig]
        model_list_ref = model_list_orig   # unquantized, single seed reference
        print(f"Quantized models with: {quant_type}\n")
    else:
        model_list_exp = model_list_orig
        model_list_ref = None

    # Real features for FID / Precision (once)
    _offload_inception()
    feats_real = _load_real_features(select_idx)

    # ── Rank comparison mode ───────────────────────────────────────────────────
    # When enabled, seeds are sorted worst→best by score_metric at each checkpoint
    # and we record metrics for ranks [1, N//4, N//2, N] as proxies for
    # "best-of-1", "best-of-N//4", "best-of-N//2", "best-of-N" experiments.
    use_rank_cmp = FLAGS.rank_comparison and num_seeds >= 4
    if use_rank_cmp:
        raw = sorted({1, num_seeds // 4, num_seeds // 2, num_seeds})
        rank_tiers = [t for t in raw if 1 <= t <= num_seeds]
        print(f"Rank-comparison mode: tiers = {rank_tiers} "
              f"(proxies for best-of-{rank_tiers} experiments)\n")

    # ── Accumulators ──────────────────────────────────────────────────────────
    # per_step_results[step][label][metric] = list of trial values
    if use_rank_cmp:
        labels_used = [f'rank_{t}' for t in rank_tiers]
    else:
        labels_used = ['single', 'multi'] + (['ref'] if quant_type != 'none' else [])
    metrics     = ['is', 'fid', 'precision']

    per_step = {
        step: {lbl: {m: [] for m in metrics} for lbl in labels_used}
        for step in eval_steps
    }
    # all_seed_scores[step][trial][seed_idx] = {'is':..,'fid':..,'precision':..}
    # Saves raw scores for every seed so nothing is discarded.
    all_seed_scores = {step: [] for step in eval_steps}
    all_trial_energies = []   # (num_trials, num_seeds)
    best_seed_indices  = []

    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{num_trials}")
        print(f"{'='*60}")

        labels = []
        for six in select_idx:
            lb = np.tile(np.eye(2)[six:six+1], (n, 1))
            labels.append(torch.Tensor(lb).cuda())

        # ── Run all seeds (quantized / default) ───────────────────────────────
        _offload_inception()
        seed_ckpts    = []   # list[dict{step: im_cpu}]
        seed_energies = []

        for seed_idx in range(num_seeds):
            seed = trial * 1000 + seed_idx
            print(f"  Seed {seed_idx+1}/{num_seeds} (rng={seed}) ...", end=' ', flush=True)
            x0 = _warmup(model_list_exp, labels, n, im_size, seed)
            ckpts, energy = _refine_checkpoints(
                model_list_exp, labels, x0, n, im_size, seed, eval_steps)
            seed_ckpts.append(ckpts)
            seed_energies.append(energy)
            print(f"energy={energy:.4f}")

        all_trial_energies.append(seed_energies)

        # ── Optionally run unquantized reference (single seed) ─────────────────
        if model_list_ref is not None:
            print("  Running unquantized reference (seed 0) ...", end=' ', flush=True)
            _offload_inception()
            seed_ref = trial * 1000  # same seed as seed_idx=0
            x0_ref = _warmup(model_list_ref, labels, n, im_size, seed_ref)
            ref_ckpts, ref_energy = _refine_checkpoints(
                model_list_ref, labels, x0_ref, n, im_size, seed_ref, eval_steps)
            print(f"energy={ref_energy:.4f}")
        else:
            ref_ckpts = None

        # ── Score each checkpoint ─────────────────────────────────────────────
        for step in eval_steps:
            # Score all seeds at this step
            seed_scores = []
            for s_ckpts in seed_ckpts:
                seed_scores.append(_score_batch(s_ckpts[step], feats_real))

            # Sort seed indices worst → best by score_metric
            if FLAGS.score_metric == 'energy':
                sort_key = seed_energies          # lower=better → worst = highest
                sorted_idx = list(np.argsort(sort_key)[::-1])
            elif FLAGS.score_metric == 'fid':
                sort_key = [s['fid'] for s in seed_scores]   # lower=better
                sorted_idx = list(np.argsort(sort_key)[::-1])
            elif FLAGS.score_metric == 'precision':
                sort_key = [s['precision'] for s in seed_scores]  # higher=better
                sorted_idx = list(np.argsort(sort_key))
            else:  # 'is'
                sort_key = [s['is'] for s in seed_scores]    # higher=better
                sorted_idx = list(np.argsort(sort_key))
            # sorted_idx[0] = worst seed, sorted_idx[-1] = best seed

            best_idx = sorted_idx[-1]

            # Save raw scores for all seeds at this step / trial
            all_seed_scores[step].append(seed_scores)   # list of num_seeds dicts

            if use_rank_cmp:
                # Record metrics for each rank tier (1-indexed from worst)
                log_parts = [f"step={step:3d}"]
                for t in rank_tiers:
                    chosen = sorted_idx[t - 1]   # t=1 → worst, t=N → best
                    lbl = f'rank_{t}'
                    for m in metrics:
                        per_step[step][lbl][m].append(seed_scores[chosen][m])
                    log_parts.append(
                        f"rank{t:2d}[seed{chosen}] "
                        f"IS={seed_scores[chosen]['is']:.3f} "
                        f"FID={seed_scores[chosen]['fid']:.1f} "
                        f"Prec={seed_scores[chosen]['precision']:.3f}")
                print("  " + " | ".join(log_parts))
            else:
                # Original single-vs-multi mode
                for m in metrics:
                    per_step[step]['single'][m].append(seed_scores[0][m])
                    per_step[step]['multi'][m].append(seed_scores[best_idx][m])

                # Reference (unquantized) if applicable
                ref_scores = None
                if ref_ckpts is not None:
                    ref_scores = _score_batch(ref_ckpts[step], feats_real)
                    for m in metrics:
                        per_step[step]['ref'][m].append(ref_scores[m])

                print(f"  step={step:3d} | "
                      f"single IS={seed_scores[0]['is']:.3f} FID={seed_scores[0]['fid']:.1f} Prec={seed_scores[0]['precision']:.3f} | "
                      f"multi  IS={seed_scores[best_idx]['is']:.3f} FID={seed_scores[best_idx]['fid']:.1f} Prec={seed_scores[best_idx]['precision']:.3f}"
                      + (f" | ref IS={ref_scores['is']:.3f} FID={ref_scores['fid']:.1f} Prec={ref_scores['precision']:.3f}"
                         if ref_scores is not None else ""))

        if len(best_seed_indices) == trial:
            # Record the best_idx from the final step
            best_seed_indices.append(best_idx)

        # Save images for trial 0
        if trial == 0:
            final_step = eval_steps[-1]
            if use_rank_cmp:
                for t in rank_tiers:
                    chosen = sorted_idx[t - 1]
                    _save_image_grid(
                        seed_ckpts[chosen][final_step], im_size,
                        osp.join(FLAGS.out_dir, f'rank_{t}_seed{chosen}_trial0.png'))
            else:
                _save_image_grid(seed_ckpts[0][final_step], im_size,
                                 osp.join(FLAGS.out_dir, 'single_seed_trial0.png'))
                _save_image_grid(seed_ckpts[best_idx][final_step], im_size,
                                 osp.join(FLAGS.out_dir, 'multi_seed_best_trial0.png'))
            _save_all_seeds_grid(
                [ckpts[final_step] for ckpts in seed_ckpts],
                im_size, best_idx,
                osp.join(FLAGS.out_dir, 'all_seeds_trial0.png'))

    # ── Summary ───────────────────────────────────────────────────────────────
    final = eval_steps[-1]
    print("\n" + "="*60 + "\nSUMMARY  (final step)\n" + "="*60)
    if use_rank_cmp:
        col_w = 12
        header = f"{'Metric':<12}" + "".join(
            f" | {'best-of-'+str(t):>{col_w}}" for t in rank_tiers)
        print(header)
        print("-" * len(header))
        for metric in metrics:
            row = f"{metric:<12}"
            for t in rank_tiers:
                row += f" | {np.mean(per_step[final][f'rank_{t}'][metric]):>{col_w}.4f}"
            print(row)
    else:
        header = f"{'Metric':<12}"
        if 'ref' in labels_used:
            header += f" | {'Ref (no quant)':>14}"
        header += f" | {f'Quantized single ({quant_type})' if quant_type!='none' else 'Single seed':>20} | {'Best of %d' % num_seeds:>10} | {'Δ (multi-single)':>16}"
        print(header)
        print("-" * len(header))
        for metric in metrics:
            row = f"{metric:<12}"
            if 'ref' in labels_used:
                row += f" | {np.mean(per_step[final]['ref'][metric]):>14.4f}"
            s = np.mean(per_step[final]['single'][metric])
            m = np.mean(per_step[final]['multi'][metric])
            d = m - s
            row += f" | {s:>20.4f} | {m:>10.4f} | {d:>+16.4f}"
            print(row)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_metric_figures(per_step, eval_steps, labels_used, num_seeds, quant_type)
    _plot_energy_heatmap(all_trial_energies, best_seed_indices, num_seeds, num_trials)

    np.save(osp.join(FLAGS.out_dir, 'multi_seed_results.npy'), {
        'per_step': per_step,
        # all_seed_scores[step][trial] = list of num_seeds dicts {is, fid, precision}
        'all_seed_scores': all_seed_scores,
        'all_trial_energies': all_trial_energies,
        'best_seed_indices':  best_seed_indices,
        'eval_steps': eval_steps,
        'num_seeds': num_seeds,
        'quant_type': quant_type,
        'select_idx': select_idx,
    })
    print(f"\nSaved all results to {FLAGS.out_dir}/")


# ── Plotting ──────────────────────────────────────────────────────────────────

_METRIC_CFG = {
    'is':        ('Inception Score',  'IS (higher = better)',  True),
    'fid':       ('FID',              'FID (lower = better)',  False),
    'precision': ('Precision',        'Precision (higher = better)', True),
}

# Colors/styles for non-rank mode
_COLOR = {
    'single': '#4c72b0',
    'multi':  '#dd8452',
    'ref':    '#55a868',
}
_LABEL = {
    'single': 'Quantized single seed',
    'multi':  'Quantized best-of-N',
    'ref':    'Default (no quant, single seed)',
}
_LABEL_NOQUANT = {
    'single': 'Single seed',
    'multi':  'Best of N seeds',
}

# Colors for rank-comparison mode (light→dark = worst→best)
_RANK_COLORS = ['#aec7e8', '#6baed6', '#2171b5', '#08306b']
_RANK_STYLES = [':', '--', '-.', '-']


def _plot_metric_figures(per_step, eval_steps, labels_used, num_seeds, quant_type):
    x = np.array(eval_steps)

    # Detect rank-comparison mode from label names
    is_rank_mode = any(lbl.startswith('rank_') for lbl in labels_used)

    for metric, (title, ylabel, higher_better) in _METRIC_CFG.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, lbl in enumerate(labels_used):
            vals   = np.array([per_step[s][lbl][metric] for s in eval_steps])  # (steps, trials)
            mean_v = vals.mean(axis=1)
            std_v  = vals.std(axis=1)

            if is_rank_mode:
                t     = int(lbl.split('_')[1])   # rank_{t}
                color = _RANK_COLORS[i % len(_RANK_COLORS)]
                ls    = _RANK_STYLES[i % len(_RANK_STYLES)]
                label = f'best-of-{t}  (rank {t}/{num_seeds})'
            else:
                color = _COLOR[lbl]
                if quant_type == 'none':
                    label = _LABEL_NOQUANT.get(lbl, lbl)
                else:
                    label = _LABEL.get(lbl, lbl)
                if lbl == 'multi':
                    label = label.replace('N', str(num_seeds))
                ls = '--' if lbl == 'single' else ('-' if lbl == 'multi' else ':')

            ax.plot(x, mean_v, ls, color=color, label=label, linewidth=2)
            ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                            color=color, alpha=0.15)

        ax.set_xlabel('MCMC step')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} over MCMC steps\n'
                     f'(sorted by: {FLAGS.score_metric}, '
                     f'{FLAGS.num_trials} trials)'
                     + (f'\n[{quant_type} quantization]' if quant_type != 'none' else ''))
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = osp.join(FLAGS.out_dir, f'{metric}_vs_steps.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def _plot_energy_heatmap(all_trial_energies, best_seed_indices, num_seeds, num_trials):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Heatmap
    arr = np.array(all_trial_energies)  # (num_trials, num_seeds)
    im  = axes[0].imshow(arr, aspect='auto', cmap='viridis_r', origin='upper')
    for t, b in enumerate(best_seed_indices):
        axes[0].plot(b, t, 'r*', markersize=10)
    axes[0].set_xlabel('Seed index')
    axes[0].set_ylabel('Trial')
    axes[0].set_title('Final energy per seed (lower=better)\nred★ = chosen best')
    plt.colorbar(im, ax=axes[0], label='Energy')

    # Histogram of chosen best indices
    axes[1].hist(best_seed_indices, bins=np.arange(-0.5, num_seeds + 0.5, 1),
                 rwidth=0.8, color='#55a868')
    axes[1].set_xlabel('Seed index chosen as best')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Best seed distribution ({num_trials} trials)')
    axes[1].set_xticks(range(num_seeds))
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = osp.join(FLAGS.out_dir, 'energy_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Image saving ──────────────────────────────────────────────────────────────

def _save_image_grid(im_cpu, im_size, path, ncols=4):
    imgs  = im_cpu.numpy()
    n     = min(16, imgs.shape[0])
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    canvas = np.ones((nrows*im_size, ncols*im_size, 3), dtype=np.uint8) * 255
    for idx in range(n):
        r, c = divmod(idx, ncols)
        canvas[r*im_size:(r+1)*im_size, c*im_size:(c+1)*im_size] = \
            (imgs[idx].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    imsave(path, canvas)


def _save_all_seeds_grid(seed_images, im_size, best_idx, path):
    num_seeds = len(seed_images)
    ncols  = min(4, seed_images[0].shape[0])
    canvas = np.ones((num_seeds*im_size, ncols*im_size, 3), dtype=np.uint8) * 200
    for s, im_cpu in enumerate(seed_images):
        imgs = im_cpu.numpy()
        for c in range(min(ncols, imgs.shape[0])):
            canvas[s*im_size:(s+1)*im_size, c*im_size:(c+1)*im_size] = \
                (imgs[c].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        if s == best_idx:
            b = 3
            canvas[s*im_size:s*im_size+b, :]         = [0,200,0]
            canvas[(s+1)*im_size-b:(s+1)*im_size, :]  = [0,200,0]
            canvas[s*im_size:(s+1)*im_size, :b]        = [0,200,0]
            canvas[s*im_size:(s+1)*im_size, -b:]       = [0,200,0]
    imsave(path, canvas)


# ── Entry point ───────────────────────────────────────────────────────────────

def combine_main(models, resume_iters, select_idx):
    model_list = []
    for model, resume_iter in zip(models, resume_iters):
        model_path = osp.join("cachedir", model, f"model_{resume_iter}.pth")
        checkpoint = torch.load(model_path)
        FLAGS_model = checkpoint['FLAGS']
        for k, v in dict(self_attn=False, multiscale=False,
                         alias=False, square_energy=False, cond=False).items():
            if not hasattr(FLAGS_model, k):
                FLAGS_model[k] = v
        m = CelebAModel(FLAGS_model)
        m.load_state_dict(checkpoint['ema_model_state_dict_0'])
        model_list.append(m.cuda().eval())

    multi_seed_inference(model_list, select_idx)


if __name__ == "__main__":
    FLAGS(sys.argv)

    models_orig      = ['celeba_128_male_2', 'celeba_128_old_2',
                        'celeba_128_smiling_2', 'celeba_128_wavy_hair_2']
    resume_iters_orig = ["latest", "6000", "13000", "9000"]

    models       = [models_orig[1], models_orig[0], models_orig[2], models_orig[3]]
    resume_iters = [resume_iters_orig[1], resume_iters_orig[0],
                    resume_iters_orig[2], resume_iters_orig[3]]
    select_idx   = [1, 1, 1, 0]

    FLAGS.step_lr = FLAGS.step_lr / len(models)
    combine_main(models, resume_iters, select_idx)
