import math
from tqdm import tqdm
import random
import copy
from absl import flags
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models_2 import CelebAModel
from utils import ReplayBuffer
import os.path as osp
import numpy as np
from scipy.misc import imsave
from torchvision import transforms
import os
from itertools import product
from PIL import Image
try:
    from classify_eval import ResNet, BasicBlock
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
import torch
from data import CelebAHQOverfit

flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 4, 'Number of workers to do things')
flags.DEFINE_string('logdir', 'cachedir', 'directory for logging')
flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_float('step_lr', 500.0, 'size of gradient descent size')
flags.DEFINE_bool('cclass', True, 'not cclass')
flags.DEFINE_bool('proj_cclass', False, 'use for backwards compatibility reasons')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_integer('num_steps', 200, 'number of steps to optimize the label')
flags.DEFINE_string('task', 'combination_figure', 'conceptcombine, combination_figure, negation_figure, or_figure, negation_eval')

flags.DEFINE_bool('eval', False, 'Whether to quantitively evaluate models')
flags.DEFINE_bool('latent_energy', False, 'latent energy in model')
flags.DEFINE_bool('proj_latent', False, 'Projection of latents')

flags.DEFINE_integer('num_samples', 64, 'Number of images generated per MCMC step (higher = more reliable FID/precision; 512 recommended)')
flags.DEFINE_integer('fid_pca_dim', 0, 'PCA dimension before FID/precision (0 = no PCA; 256 recommended when num_samples >= 512)')
flags.DEFINE_string('out_dir', 'quant_results', 'Directory to save output images, plots, and metrics')
flags.DEFINE_bool('cross_step', False, 'Whether to also run cross-step inference (use g[t-2] instead of g[t-1])')


# Whether to train for gentest
flags.DEFINE_bool('train', False, 'whether to train on generalization into multiple different predictions')

FLAGS = flags.FLAGS


def conceptcombineeval(model_list, select_idx):
    dataset = CelebAHQOverfit()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4)

    n = 64
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)


    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    gt_ims = []
    fake_ims = []

    for data, label in dataloader:
        gt_ims.extend(list((data.numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)))

        im = torch.rand(n, 3, 128, 128).cuda()
        im_noise = torch.randn_like(im).detach()
        # First get good initializations for sampling
        for i in range(10):
            for i in range(20):
                im_noise.normal_()
                im = im + 0.001 * im_noise
                # im.requires_grad = True
                im.requires_grad_(requires_grad=True)
                energy = 0

                for model, label in zip(model_list, labels):
                    energy = model.forward(im, label) +  energy

                # print("step: ", i, energy.mean())
                im_grad = torch.autograd.grad([energy.sum()], [im])[0]

                im = im - FLAGS.step_lr * im_grad
                im = im.detach()

                im = torch.clamp(im, 0, 1)

            im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
            im = (im * 255).astype(np.uint8)

            ims = []
            for i in range(im.shape[0]):
                im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
                ims.append(im_i)

            im = torch.Tensor(np.array(ims)).cuda()

        # Then refine the images

        for i in range(FLAGS.num_steps):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            for model, label in zip(model_list, labels):
                energy = model.forward(im, label) +  energy

            # print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach().cpu()
        fake_ims.extend(list((fake_images.numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)))
        if len(gt_ims) > 100:
            break



    get_fid_score(gt_ims, fake_ims)
    fake_ims = np.array(fake_ims)
    fake_ims_flat = fake_ims.reshape(fake_ims.shape[0], -1)
    std_im = np.std(fake_ims, axis=0).mean()
    print("standard deviation of image", std_im)
    import pdb
    pdb.set_trace()
    print("here")



def conceptcombine(model_list, select_idx):

    n = 64
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)

    im = torch.rand(n, 3, 128, 128).cuda()
    im_noise = torch.randn_like(im).detach()

    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    # First get good initializations for sampling
    for i in range(10):
        for i in range(20):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            for model, label in zip(model_list, labels):
                energy = model.forward(im, label) +  energy

            # print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
        im = (im * 255).astype(np.uint8)

        ims = []
        for i in range(im.shape[0]):
            im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
            ims.append(im_i)

        im = torch.Tensor(np.array(ims)).cuda()

    # Then refine the images

    for i in range(FLAGS.num_steps):
        im_noise.normal_()
        im = im + 0.001 * im_noise
        # im.requires_grad = True
        im.requires_grad_(requires_grad=True)
        energy = 0

        for model, label in zip(model_list, labels):
            energy = model.forward(im, label) +  energy

        print("step: ", i, energy.mean())
        im_grad = torch.autograd.grad([energy.sum()], [im])[0]

        im = im - FLAGS.step_lr * im_grad
        im = im.detach()

        im = torch.clamp(im, 0, 1)

    output = im.detach().cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = output.reshape((-1, 8, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * 8, 3))
    imsave("debug.png", output)


def quant_eval(model_list, select_idx):
    classify_model = ResNet(BasicBlock, [2, 2, 2], num_classes=4)
    classify_model = classify_model.cuda().eval()
    classify_model.load_state_dict(torch.load('celeba_classify.t')['celeba_classify_dict'])
    classify_model = classify_model.eval()
    young = []
    female = []
    smiling = []
    wavy = []

    n = 16


    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()
    im_size = 128
    transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])

    for i in range(30):
        n = 16

        labels_val = [random.randint(0, 1) for _ in range(4)]
        nelem = len(select_idx)
        labels_val = select_idx[:nelem] + labels_val[nelem:]
        # print(labels_val)
        labels = []

        for six in labels_val:
            label_ix = np.eye(2)[six]
            label_batch = np.tile(label_ix[None, :], (n, 1))
            label = torch.Tensor(label_batch).cuda()
            labels.append(label)

        im = torch.rand(n, 3, 128, 128).cuda()
        im.requires_grad_()
        im_noise = torch.randn_like(im).detach()
        # First get good initializations for sampling
        for i in range(10):
            for i in range(20):
                im_noise.normal_()
                im = im + 0.001 * im_noise
                im.requires_grad_(requires_grad=True)
                energy = 0

                for model, label in zip(model_list, labels):
                    energy = model.forward(im, label) +  energy

                # print("step: ", i, energy.mean())
                im_grad = torch.autograd.grad([energy.sum()], [im])[0]

                im = im - FLAGS.step_lr * im_grad
                im = im.detach()

                im = torch.clamp(im, 0, 1)

            im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
            im = (im * 255).astype(np.uint8)

            ims = []
            for i in range(im.shape[0]):
                im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
                ims.append(im_i)

            im = torch.Tensor(np.array(ims)).cuda()

        # Then refine the images

        for i in range(FLAGS.num_steps):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            # im.requires_grad = True
            im.requires_grad_(requires_grad=True)
            energy = 0

            for counter, (model, label) in enumerate(zip(model_list, labels)):
                # print(counter)
                energy = model.forward(im, label) +  energy

            # print("step: ", i, energy.mean())
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]

            im = im - FLAGS.step_lr * im_grad
            im = im.detach()

            im = torch.clamp(im, 0, 1)

        im = im.detach()
        # im = im * 255 / 256 + torch.rand_like(im) * 1. / 256.

        # output = im.detach().cpu().numpy()
        # output = output.transpose((0, 2, 3, 1))
        # output = output.reshape((-1, 8, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * 8, 3))
        # imsave("debug.png", output)
        # import pdb
        # pdb.set_trace()

        pred = classify_model.forward(im)
        attribute_label = pred.detach().cpu().numpy()

        for at in attribute_label:
        # Pred has the form (smiling, male)
            if labels_val[0] == 1:
                young.append(at[0] > 0.5)
            else:
                young.append(at[0] < 0.5)

            if labels_val[1] == 1:
                female.append(at[1] > 0.1)
            else:
                female.append(at[1] < 0.1)

            if labels_val[2] == 1:
                smiling.append(at[2] > 0.5)
            else:
                smiling.append(at[2] < 0.5)

            if labels_val[3] == 1:
                wavy.append(at[3] > 0.5)
            else:
                wavy.append(at[3] < 0.5)

            # young.append(at[0] > 0.5)
            # female.append(at[1] < 0.5)
            # smiling.append(at[2] > 0.5)
            # wavy.append(at[3] > 0.5)

        print("Young mean ", np.mean(young))
        print("Female mean ", np.mean(female))
        print("Smiling mean ", np.mean(smiling))
        print("Wavy mean ", np.mean(wavy))

def combination_figure(sess, kvs, select_idx):
    n = 64

    print("here")
    labels = kvs['labels']
    x_mod = kvs['x_mod']
    X_NOISE = kvs['X_NOISE']
    model_base = kvs['model_base']
    weights = kvs['weights']
    feed_dict = {}

    for i, label in enumerate(labels):
        j = select_idx[i]
        feed_dict[label] = np.tile(np.eye(2)[j:j+1], (16, 1))

    x_noise = np.random.uniform(0, 1, (n, 128, 128, 3))
    # x_noise =  np.random.uniform(0, 1, (n, 128, 128, 3)) / 2 + np.random.uniform(0, 1, (n, 1, 1, 3)) * 1. / 2

    feed_dict[X_NOISE] = x_noise

    output = sess.run([x_mod], feed_dict)[0]
    output = output.reshape((n * 128, 128, 3))
    imsave("debug.png", output)


def negation_figure(sess, kvs, select_idx):
    n = 64

    labels = kvs['labels']
    x_mod = kvs['x_mod']
    X_NOISE = kvs['X_NOISE']
    model_base = kvs['model_base']
    weights = kvs['weights']
    feed_dict = {}

    for i, label in enumerate(labels):
        j = select_idx[i]
        feed_dict[label] = np.tile(np.eye(2)[j:j+1], (n, 1))

    x_noise = np.random.uniform(0, 1, (n, 128, 128, 3))
    feed_dict[X_NOISE] = x_noise

    output = sess.run([x_mod], feed_dict)[0]
    output = output.reshape((n * 128, 128, 3))
    imsave("debug.png", output)


def quantize_model(model, quant_type):
    """Replace all nn.Linear layers with quantized versions.

    Supported quant_type values:
      - 'nf4': bitsandbytes 4-bit NormalFloat
      - 'fp4': bitsandbytes 4-bit FloatingPoint
      - 'int8': bitsandbytes 8-bit LLM.int8()
      - 'fp8': simulate FP8 weight-only via torch.float8_e4m3fn cast
    """
    import bitsandbytes as bnb
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            has_bias = module.bias is not None

            if quant_type in ('nf4', 'fp4'):
                quant_linear = bnb.nn.Linear4bit(
                    in_f, out_f, bias=has_bias,
                    quant_type=quant_type
                )
                quant_linear.weight = bnb.nn.Params4bit(
                    module.weight.data.clone(), requires_grad=False, quant_type=quant_type
                )
                if has_bias:
                    quant_linear.bias = nn.Parameter(module.bias.data.clone())
                quant_linear = quant_linear.cuda()  # triggers weight quantization + sets quant_state
                setattr(model, name, quant_linear)

            elif quant_type == 'int8':
                # Simulate int8 weight-only quantization via symmetric per-tensor scaling.
                # bitsandbytes Linear8bitLt uses LLM.int8() which requires a specific
                # batch size for backward and breaks gradient-based MCMC sampling.
                w = module.weight.data.float()
                scale = 127.0 / (w.abs().max().clamp(min=1e-12))
                w_int8 = (w * scale).round().clamp(-128, 127)
                module.weight.data = (w_int8 / scale).to(module.weight.data.dtype)
                # Module stays as nn.Linear — no structural change needed.

            elif quant_type == 'fp8':
                # Simulate FP8 E4M3 weight-only quantization for PyTorch < 2.1.
                # E4M3 range: max representable value = 448.0, 3 mantissa bits → 8 levels per exponent.
                w = module.weight.data.float()
                fp8_max = 448.0
                scale = fp8_max / (w.abs().max().clamp(min=1e-12))
                w_scaled = (w * scale).clamp(-fp8_max, fp8_max)
                # Round to nearest 1/8 (3 mantissa bits → resolution of 2^-3 at each exponent)
                w_quantized = (w_scaled * 8).round() / 8
                module.weight.data = (w_quantized / scale).to(module.weight.data.dtype)
                # Module stays as nn.Linear — no structural change needed.
            else:
                raise ValueError(f"Unknown quant_type: {quant_type}")
        else:
            quantize_model(module, quant_type)
    return model


def _run_trajectory(model_list, labels, x0, n, im_size):
    """Run one full MCMC refinement trajectory from x0. Returns list of CPU tensors."""
    trajectory = []
    im = x0.clone()
    for step in range(FLAGS.num_steps):
        g = torch.Generator(device='cuda')
        g.manual_seed(1000 + step)
        noise = torch.empty(n, 3, im_size, im_size, device='cuda').normal_(generator=g)
        im = im + 0.001 * noise
        im.requires_grad_(True)
        im_grad = None
        for m, l in zip(model_list, labels):
            g = torch.autograd.grad([m.forward(im, l).sum()], [im])[0]
            im_grad = g if im_grad is None else im_grad + g
        im = (im - FLAGS.step_lr * im_grad).detach().clamp(0, 1)
        trajectory.append(im.cpu())
    return trajectory


def _run_trajectory_cross_step(model_list_orig, labels, x0, n, im_size, quant_type=None):
    """Run MCMC with cross-step gradient: use g[t-2] to update sample before fwd/bwd.

    Steps 0 and 1 run normally (no two-step-earlier gradient available).
    From step 2 onward:
        im = im + noise - step_lr * g[t-2]   (apply stale gradient before fwd)
        g[t] = compute_grad(im)               (compute current gradient for future use)

    Args:
        model_list_orig: list of models (will be deep-copied if quant_type is set)
        quant_type: optional quantization type ('fp4', 'fp8', 'int8', 'nf4').
                    If given, models are deep-copied and quantized internally.
    Returns:
        List of CPU tensors (one per step).
    """
    if quant_type is not None:
        model_list = [quantize_model(copy.deepcopy(m), quant_type) for m in model_list_orig]
    else:
        model_list = model_list_orig

    trajectory = []
    im = x0.clone()
    grad_prev2 = None  # g[t-2]
    grad_prev1 = None  # g[t-1]

    for step in range(FLAGS.num_steps):
        g_gen = torch.Generator(device='cuda')
        g_gen.manual_seed(1000 + step)
        noise = torch.empty(n, 3, im_size, im_size, device='cuda').normal_(generator=g_gen)
        im = im + 0.001 * noise

        if step >= 2:
            # Cross-step: apply gradient from two steps ago before the forward pass
            im = (im - FLAGS.step_lr * grad_prev2).clamp(0, 1)

        im.requires_grad_(True)
        im_grad = None
        for m, l in zip(model_list, labels):
            g = torch.autograd.grad([m.forward(im, l).sum()], [im])[0]
            im_grad = g if im_grad is None else im_grad + g

        if step < 2:
            # Normal update for initial two steps (no stale gradient yet)
            im = (im - FLAGS.step_lr * im_grad).detach().clamp(0, 1)
        else:
            im = im.detach().clamp(0, 1)

        # Advance gradient history ring buffer
        grad_prev2 = grad_prev1
        grad_prev1 = im_grad.detach()

        trajectory.append(im.cpu())

    return trajectory


_inception_model = None
_inception_logits_model = None


def _get_inception_features_model():
    """Inception v3 with the final pool layer as output (2048-d), for FID/precision."""
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3
        m = inception_v3(pretrained=True, transform_input=False)
        m.fc = nn.Identity()
        _inception_model = m.cuda().eval()
    else:
        _inception_model.cuda()
    return _inception_model


def _extract_features(images_cuda, batch_size=16):
    """Extract 2048-d Inception pool features. images_cuda: (N,3,H,W) in [0,1]."""
    import torch.nn.functional as F
    from torchvision import transforms as T
    model = _get_inception_features_model()
    resize = T.Resize((299, 299), antialias=True)
    feats = []
    with torch.no_grad():
        for i in range(0, images_cuda.shape[0], batch_size):
            x = resize(images_cuda[i:i+batch_size])
            out = model(x)
            if hasattr(out, 'logits'):
                out = out.logits
            feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()  # (N, 2048)


def _maybe_pca(feats_real, feats_fake):
    """Optionally PCA-project features. Controlled by --fid_pca_dim flag."""
    pca_dim = FLAGS.fid_pca_dim
    if pca_dim <= 0:
        return feats_real, feats_fake
    from sklearn.decomposition import PCA
    n_components = min(pca_dim, feats_real.shape[0] - 1, feats_fake.shape[0] - 1, feats_real.shape[1])
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components)
    pca.fit(np.concatenate([feats_real, feats_fake], axis=0))
    return pca.transform(feats_real), pca.transform(feats_fake)


def _compute_fid(feats_real, feats_fake):
    """Fréchet Inception Distance between two sets of features."""
    from scipy import linalg
    r, f = _maybe_pca(feats_real, feats_fake)
    mu_r, sigma_r = r.mean(0), np.cov(r, rowvar=False)
    mu_f, sigma_f = f.mean(0), np.cov(f, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


def _compute_precision(feats_real, feats_fake, k=3):
    """Precision: fraction of fake samples within the real manifold (k-NN)."""
    from sklearn.neighbors import NearestNeighbors
    r, f = _maybe_pca(feats_real, feats_fake)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(r)
    dists_real, _ = nbrs.kneighbors(r)
    radii = dists_real[:, -1]
    nearest_real_dist = nbrs.kneighbors(f, n_neighbors=1, return_distance=True)[0][:, 0]
    nearest_real_idx  = nbrs.kneighbors(f, n_neighbors=1, return_distance=False)[:, 0]
    precision = float((nearest_real_dist <= radii[nearest_real_idx]).mean())
    return precision


def _get_inception():
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3
        m = inception_v3(pretrained=True, transform_input=False)
        m.fc = nn.Identity()   # extract 2048-d pool features, not logits
        m = m.cuda().eval()
        _inception_model = m
    else:
        _inception_model.cuda()
    return _inception_model


def _compute_inception_score(im_gpu):
    """Inception Score proxy: mean entropy of softmax predictions.

    Uses torchvision's pretrained Inception v3. Images are resized to 299×299
    as required by Inception. Returns mean KL(p(y|x) || p(y)), i.e. the IS
    contribution per sample — higher means more confident and diverse predictions.
    """
    import torch.nn.functional as F
    from torchvision import transforms as T
    from torchvision.models import inception_v3

    # Build a temporary Inception with logits output for IS
    model = inception_v3(pretrained=True, transform_input=False).cuda().eval()
    resize = T.Resize((299, 299), antialias=True)
    with torch.no_grad():
        x = resize(im_gpu)
        logits = model(x)
        if hasattr(logits, 'logits'):  # InceptionOutputs namedtuple
            logits = logits.logits
        p_yx = F.softmax(logits, dim=1)            # (n, 1000)
        p_y = p_yx.mean(dim=0, keepdim=True)       # (1, 1000) marginal
        kl = (p_yx * (p_yx.clamp(min=1e-8).log() - p_y.clamp(min=1e-8).log())).sum(dim=1)
    return kl.mean().item()


def quantization_figure(model_list_orig, select_idx, run_cross_step=False):
    """Run default + 4 quantized trajectories from the same x0 and plot comparisons.

    Flow:
      1. Warm-up with default models to produce a shared x0.
      2. Run default trajectory fully; store each step's image on CPU.
      3. For each quant_type in [fp4, fp8, int8, nf4]:
           - Run full trajectory from the same x0.
           - At each step t compute:
               L2 divergence = ||x_t_quant - x_t_default||_2 / ||x_t_default||_2
               accuracy of x_t_quant (and x_t_default, already recorded).
      4. Save per-step metrics and comparison plots.

    All trajectories use the same deterministic noise at each step so the only
    difference between them is the model weights.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    quant_types = ['fp4', 'fp8', 'int8', 'nf4']
    n = FLAGS.num_samples  # set via --num_samples (64=fast/noisy, 512=recommended, 1024=reliable)
    im_size = 128
    fid_interval = 5  # compute FID/precision every this many steps

    # --- Real image features for FID and precision ---
    # Filter real images to match the SAME attributes as select_idx so the
    # reference distribution is comparable to the generated distribution.
    # Model order: [old, male, smiling, wavy_hair] → CelebA column indices (0-based):
    #   old      = Young == -1  → col 39
    #   male     = Male         → col 20
    #   smiling  = Smiling      → col 31
    #   wavy_hair= Wavy_Hair    → col 33
    # select_idx value 1 → attribute present (+1 in CelebA)
    # select_idx value 0 → attribute absent  (-1 in CelebA)
    import pandas as pd
    from torchvision import transforms as T
    attr_cols   = [39, 20, 31, 33]          # col indices matching model order
    attr_signs  = [(-1, 1), (1, -1), (1, -1), (-1, 1)]  # (value when select=1, value when select=0)
    print("=== Filtering real images by attributes and extracting features for FID/precision ===")
    celeba_dir  = "data/celeba/img_align_celeba"
    attr_df = pd.read_csv("data/celeba/list_attr_celeba.txt", sep=r"\s+", skiprows=1)
    mask = np.ones(len(attr_df), dtype=bool)
    for col_idx, (val1, val0), six in zip(attr_cols, attr_signs, select_idx):
        col_name = attr_df.columns[col_idx]
        expected = val1 if six == 1 else val0
        mask &= (attr_df[col_name].values == expected)
    filtered_files = attr_df.index[mask].tolist()
    print(f"Found {len(filtered_files)} real images matching attributes")
    np.random.seed(0)
    chosen = np.random.choice(filtered_files, size=min(2000, len(filtered_files)), replace=False)
    to_tensor = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    real_imgs = []
    for fname in chosen:
        img = Image.open(osp.join(celeba_dir, fname)).convert('RGB')
        real_imgs.append(to_tensor(img))
    real_imgs = torch.stack(real_imgs).cuda()
    feats_real = _extract_features(real_imgs)
    print(f"Real features extracted: {feats_real.shape} ({len(chosen)} attribute-matched images)\n")
    del real_imgs
    torch.cuda.empty_cache()

    # --- Labels ---
    labels = []
    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        labels.append(torch.Tensor(label_batch).cuda())

    # --- Augmentation transform for warm-up ---
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Compose([rnd_color_jitter, rnd_gray]),
        transforms.ToTensor(),
    ])

    # Move Inception off GPU during MCMC phases to free ~90 MB
    if _inception_model is not None:
        _inception_model.cpu()
        torch.cuda.empty_cache()

    # --- Step 1: Shared x0 via warm-up with default models ---
    print("=== Warm-up (default models) → x0 ===")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    im = torch.rand(n, 3, im_size, im_size).cuda()
    im_noise = torch.randn_like(im).detach()
    for _ in range(10):
        for _ in range(20):
            im_noise.normal_()
            im = im + 0.001 * im_noise
            im.requires_grad_(True)
            im_grad = None
            for m, l in zip(model_list_orig, labels):
                g = torch.autograd.grad([m.forward(im, l).sum()], [im])[0]
                im_grad = g if im_grad is None else im_grad + g
            im = (im - FLAGS.step_lr * im_grad).detach().clamp(0, 1)
        im_np = (im.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        ims_aug = [np.array(transform(Image.fromarray(im_np[i]))) for i in range(n)]
        im = torch.tensor(np.array(ims_aug)).cuda()
    x0 = im.detach().clone()
    print("x0 ready.\n")

    # --- Step 2: Default trajectory ---
    print("=== Running: default ===")
    model_list_default = [copy.deepcopy(m) for m in model_list_orig]
    default_traj = _run_trajectory(model_list_default, labels, x0, n, im_size)
    del model_list_default
    torch.cuda.empty_cache()

    default_acc = []
    default_fid = []
    default_precision = []
    for step, t in enumerate(default_traj):
        t_gpu = t.cuda()
        default_acc.append(_compute_inception_score(t_gpu))
        if step % fid_interval == 0 or step == len(default_traj) - 1:
            feats = _extract_features(t_gpu)
            default_fid.append(_compute_fid(feats_real, feats))
            default_precision.append(_compute_precision(feats_real, feats))
        else:
            default_fid.append(None)
            default_precision.append(None)
    print(f"Default done. Final IS: {default_acc[-1]:.4f} | FID: {default_fid[-1]:.2f} | Precision: {default_precision[-1]:.4f}\n")

    # --- Step 3: Each quantized trajectory ---
    all_results = {}   # key -> {'l2_div': [...], 'accuracy': [...], 'fid': [...], 'precision': [...]}
    final_frames = {}  # key -> last CPU tensor (for image saving)

    for qt in quant_types:
        print(f"=== Running: {qt} ===")
        if _inception_model is not None:
            _inception_model.cpu()
            torch.cuda.empty_cache()
        model_list_q = [quantize_model(copy.deepcopy(m), qt) for m in model_list_orig]
        traj = _run_trajectory(model_list_q, labels, x0, n, im_size)
        del model_list_q
        torch.cuda.empty_cache()

        l2_divs, accs, fids, precisions = [], [], [], []
        for step, (t_q, t_d) in enumerate(zip(traj, default_traj)):
            t_gpu = t_q.cuda()
            l2_divs.append(((t_q - t_d).norm(p=2) / (t_d.norm(p=2) + 1e-8)).item())
            accs.append(_compute_inception_score(t_gpu))
            if step % fid_interval == 0 or step == len(traj) - 1:
                feats = _extract_features(t_gpu)
                fids.append(_compute_fid(feats_real, feats))
                precisions.append(_compute_precision(feats_real, feats))
            else:
                fids.append(None)
                precisions.append(None)

        all_results[qt] = {'l2_div': l2_divs, 'accuracy': accs, 'fid': fids, 'precision': precisions}
        final_frames[qt] = traj[-1]
        print(f"{qt} done. Final L2: {l2_divs[-1]:.4f} | IS: {accs[-1]:.4f} | FID: {fids[-1]:.2f} | Precision: {precisions[-1]:.4f}\n")

    # --- Step 3b (optional): Cross-step trajectories ---
    cross_step_keys = []  # tracks which cross-step keys were added
    if run_cross_step:
        # Default model, cross-step
        print("=== Running: cross_step (default) ===")
        if _inception_model is not None:
            _inception_model.cpu()
            torch.cuda.empty_cache()
        cs_traj = _run_trajectory_cross_step(model_list_orig, labels, x0, n, im_size)
        torch.cuda.empty_cache()

        l2_divs, accs, fids, precisions = [], [], [], []
        for step, (t_cs, t_d) in enumerate(zip(cs_traj, default_traj)):
            t_gpu = t_cs.cuda()
            l2_divs.append(((t_cs - t_d).norm(p=2) / (t_d.norm(p=2) + 1e-8)).item())
            accs.append(_compute_inception_score(t_gpu))
            if step % fid_interval == 0 or step == len(cs_traj) - 1:
                feats = _extract_features(t_gpu)
                fids.append(_compute_fid(feats_real, feats))
                precisions.append(_compute_precision(feats_real, feats))
            else:
                fids.append(None)
                precisions.append(None)

        all_results['cross_step'] = {'l2_div': l2_divs, 'accuracy': accs, 'fid': fids, 'precision': precisions}
        final_frames['cross_step'] = cs_traj[-1]
        cross_step_keys.append('cross_step')
        print(f"cross_step done. Final L2: {l2_divs[-1]:.4f} | IS: {accs[-1]:.4f} | FID: {fids[-1]:.2f} | Precision: {precisions[-1]:.4f}\n")

        # Cross-step with each quantization type
        for qt in quant_types:
            key = f'cross_step_{qt}'
            print(f"=== Running: {key} ===")
            if _inception_model is not None:
                _inception_model.cpu()
                torch.cuda.empty_cache()
            cs_traj_q = _run_trajectory_cross_step(model_list_orig, labels, x0, n, im_size, quant_type=qt)
            torch.cuda.empty_cache()

            l2_divs, accs, fids, precisions = [], [], [], []
            for step, (t_cs, t_d) in enumerate(zip(cs_traj_q, default_traj)):
                t_gpu = t_cs.cuda()
                l2_divs.append(((t_cs - t_d).norm(p=2) / (t_d.norm(p=2) + 1e-8)).item())
                accs.append(_compute_inception_score(t_gpu))
                if step % fid_interval == 0 or step == len(cs_traj_q) - 1:
                    feats = _extract_features(t_gpu)
                    fids.append(_compute_fid(feats_real, feats))
                    precisions.append(_compute_precision(feats_real, feats))
                else:
                    fids.append(None)
                    precisions.append(None)

            all_results[key] = {'l2_div': l2_divs, 'accuracy': accs, 'fid': fids, 'precision': precisions}
            final_frames[key] = cs_traj_q[-1]
            cross_step_keys.append(key)
            print(f"{key} done. Final L2: {l2_divs[-1]:.4f} | IS: {accs[-1]:.4f} | FID: {fids[-1]:.2f} | Precision: {precisions[-1]:.4f}\n")

    # --- Summary table ---
    final_fid_def = next(v for v in reversed(default_fid) if v is not None)
    final_prec_def = next(v for v in reversed(default_precision) if v is not None)
    all_exp_keys = quant_types + cross_step_keys
    col_w = max(14, max((len(k) for k in all_exp_keys), default=0) + 2)
    print(f"\n{'':<{col_w}} | {'Avg L2 div':>10} | {'Final L2':>8} | {'Avg IS':>7} | {'Final FID':>9} | {'Final Prec':>10}")
    print("-" * (col_w + 56))
    print(f"{'default':<{col_w}} | {'N/A':>10} | {'N/A':>8} | {np.mean(default_acc):>7.4f} | {final_fid_def:>9.2f} | {final_prec_def:>10.4f}")
    for k in all_exp_keys:
        r = all_results[k]
        final_fid = next(v for v in reversed(r['fid']) if v is not None)
        final_prec = next(v for v in reversed(r['precision']) if v is not None)
        print(f"{k:<{col_w}} | {np.mean(r['l2_div']):>10.4f} | {r['l2_div'][-1]:>8.4f} "
              f"| {np.mean(r['accuracy']):>7.4f} | {final_fid:>9.2f} | {final_prec:>10.4f}")

    save_data = {
        'default_acc': default_acc, 'default_fid': default_fid, 'default_precision': default_precision,
        **all_results
    }

    # --- Step 4: Plots (only non-None FID/precision steps) ---
    steps = list(range(FLAGS.num_steps))
    fid_steps = [s for s in steps if s % fid_interval == 0 or s == FLAGS.num_steps - 1]
    colors = {'fp4': 'tab:blue', 'fp8': 'tab:orange', 'int8': 'tab:green', 'nf4': 'tab:red',
              'cross_step': 'tab:purple',
              'cross_step_fp4': 'tab:cyan', 'cross_step_fp8': 'tab:olive',
              'cross_step_int8': 'tab:brown', 'cross_step_nf4': 'tab:pink'}

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Plot 1: L2 divergence (every step)
    ax = axes[0]
    ax.axhline(0, color='black', linewidth=2.5, linestyle='--', label='default', zorder=10)
    for k in all_exp_keys:
        ls = ':' if k.startswith('cross_step') else '-'
        ax.plot(steps, all_results[k]['l2_div'], label=k, color=colors.get(k, None), linestyle=ls)
    ax.set_xlabel('MCMC step')
    ax.set_ylabel('Normalized L2 divergence')
    ax.set_title('L2 divergence vs default trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Inception score (every step)
    ax = axes[1]
    for k in all_exp_keys:
        ls = ':' if k.startswith('cross_step') else '-'
        ax.plot(steps, all_results[k]['accuracy'], label=k, color=colors.get(k, None), linestyle=ls, alpha=0.8)
    ax.plot(steps, default_acc, label='default', color='black', linewidth=2.5, linestyle='--', zorder=10)
    ax.set_xlabel('MCMC step')
    ax.set_ylabel('Inception score (IS)')
    ax.set_title('Inception score over MCMC steps (higher = more realistic & diverse)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: FID (every fid_interval steps)
    ax = axes[2]
    for k in all_exp_keys:
        ls = ':' if k.startswith('cross_step') else '-'
        ax.plot(fid_steps, [all_results[k]['fid'][s] for s in fid_steps],
                label=k, color=colors.get(k, None), linestyle=ls, alpha=0.8)
    ax.plot(fid_steps, [default_fid[s] for s in fid_steps], label='default', color='black', linewidth=2.5, linestyle='--', zorder=10)
    ax.set_xlabel('MCMC step')
    ax.set_ylabel('FID (lower = better)')
    ax.set_title(f'FID vs real CelebA (every {fid_interval} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Precision (every fid_interval steps)
    ax = axes[3]
    for k in all_exp_keys:
        ls = ':' if k.startswith('cross_step') else '-'
        ax.plot(fid_steps, [all_results[k]['precision'][s] for s in fid_steps],
                label=k, color=colors.get(k, None), linestyle=ls, alpha=0.8)
    ax.plot(fid_steps, [default_precision[s] for s in fid_steps], label='default', color='black', linewidth=2.5, linestyle='--', zorder=10)
    ax.set_xlabel('MCMC step')
    ax.set_ylabel('Precision (higher = better)')
    ax.set_title(f'Precision vs real CelebA (every {fid_interval} steps)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = FLAGS.out_dir
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(osp.join(out_dir, 'quant_comparison.png'), dpi=150)
    print(f"Saved comparison plot to {out_dir}/quant_comparison.png")

    # Save final images (8×8 grid) for default and each quant type
    def save_grid(frame_cpu, name):
        out = frame_cpu[:64].numpy().transpose(0, 2, 3, 1)
        out = out.reshape(8, 8, im_size, im_size, 3).transpose(0, 2, 1, 3, 4).reshape(8 * im_size, 8 * im_size, 3)
        imsave(osp.join(out_dir, f"quant_{name}.png"), (out * 255).astype(np.uint8))

    save_grid(default_traj[-1], 'default')
    for k in all_exp_keys:
        save_grid(final_frames[k], k)
        print(f"Saved {out_dir}/quant_{k}.png")

    np.save(osp.join(out_dir, 'quant_results.npy'), save_data)
    print(f"Saved per-step metrics to {out_dir}/quant_results.npy")

    return save_data


def combine_main(models, resume_iters, select_idx):

    model_list = []

    for model, resume_iter in zip(models, resume_iters):
        model_path = osp.join("cachedir", model, "model_{}.pth".format(resume_iter))
        checkpoint = torch.load(model_path)
        FLAGS_model = checkpoint['FLAGS']
        defaults = dict(self_attn=False, multiscale=False,
                        alias=False, square_energy=False, cond=False)
        for k, v in defaults.items():
            if not hasattr(FLAGS_model, k):
                FLAGS_model[k] = v
        model_base = CelebAModel(FLAGS_model)
        model_base.load_state_dict(checkpoint['ema_model_state_dict_0'])
        model_base = model_base.cuda()
        model_list.append(model_base)

    quantization_figure(model_list, select_idx, run_cross_step=FLAGS.cross_step)


if __name__ == "__main__":
    import sys
    FLAGS(sys.argv)

    models_orig = ['celeba_128_male_2', 'celeba_128_old_2', 'celeba_128_smiling_2', 'celeba_128_wavy_hair_2']
    resume_iters_orig = ["latest", "6000", "13000", "9000"]

    ##################################
    # Settings for the composition_figure
    models = []
    resume_iters = []
    select_idx = []
    models = [models_orig[1]]
    resume_iters = [resume_iters_orig[1]]
    select_idx = [1]

    models = models + [models_orig[0]]
    resume_iters = resume_iters + [resume_iters_orig[0]]
    select_idx = select_idx + [1]

    models = models + [models_orig[2]]
    resume_iters = resume_iters + [resume_iters_orig[2]]
    select_idx = select_idx + [1]

    models = models + [models_orig[3]]
    resume_iters = resume_iters + [resume_iters_orig[3]]
    select_idx = select_idx + [0]

    FLAGS.step_lr = FLAGS.step_lr / len(models)

    # List of 4 attributes that might be good
    # Young -> Female -> Smiling -> Wavy
    combine_main(models, resume_iters, select_idx)

