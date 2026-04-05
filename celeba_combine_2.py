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
        energy = sum(m.forward(im, l) for m, l in zip(model_list, labels))
        im_grad = torch.autograd.grad([energy.sum()], [im])[0]
        im = (im - FLAGS.step_lr * im_grad).detach().clamp(0, 1)
        trajectory.append(im.cpu())
    return trajectory


def _compute_energy_score(model_list, labels, im_gpu):
    """Return mean negative energy (higher = sample fits the model better).

    Used as a proxy for accuracy when no external classifier is available.
    The EBM assigns low energy to samples it considers high quality, so
    -energy is a natural quality score.
    """
    with torch.no_grad():
        energy = sum(m.forward(im_gpu, l) for m, l in zip(model_list, labels))
    return (-energy.mean()).item()


def quantization_figure(model_list_orig, select_idx):
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
    n = 16
    im_size = 128

    # Classifier is optional; energy score is used as fallback accuracy proxy
    thresholds = [0.5, 0.1, 0.5, 0.5]  # (old, male, smiling, wavy_hair) — unused without classifier

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
            energy = sum(m.forward(im, l) for m, l in zip(model_list_orig, labels))
            im_grad = torch.autograd.grad([energy.sum()], [im])[0]
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

    default_acc = [
        _compute_energy_score(model_list_default, labels, t.cuda())
        for t in default_traj
    ]
    print(f"Default done. Final energy score: {default_acc[-1]:.4f}\n")

    # --- Step 3: Each quantized trajectory ---
    all_results = {}   # quant_type -> {'l2_div': [...], 'accuracy': [...]}
    final_frames = {}  # quant_type -> last CPU tensor (for image saving)

    for qt in quant_types:
        print(f"=== Running: {qt} ===")
        model_list_q = [quantize_model(copy.deepcopy(m), qt) for m in model_list_orig]
        traj = _run_trajectory(model_list_q, labels, x0, n, im_size)

        l2_divs = []
        accs = []
        for t_q, t_d in zip(traj, default_traj):
            l2 = ((t_q - t_d).norm(p=2) / (t_d.norm(p=2) + 1e-8)).item()
            l2_divs.append(l2)
            accs.append(_compute_energy_score(model_list_q, labels, t_q.cuda()))

        all_results[qt] = {'l2_div': l2_divs, 'accuracy': accs}
        final_frames[qt] = traj[-1]
        print(f"{qt} done. Final L2 div: {l2_divs[-1]:.4f}, Final energy score: {accs[-1]:.4f}\n")

    # --- Summary table ---
    print(f"{'Experiment':<10} | {'Avg L2 div':>10} | {'Final L2':>8} | {'Avg score':>9} | {'Final score':>11}")
    print("-" * 60)
    print(f"{'default':<10} | {'N/A':>10} | {'N/A':>8} | {np.mean(default_acc):>9.4f} | {default_acc[-1]:>11.4f}")
    for qt in quant_types:
        r = all_results[qt]
        print(f"{qt:<10} | {np.mean(r['l2_div']):>10.4f} | {r['l2_div'][-1]:>8.4f} "
              f"| {np.mean(r['accuracy']):>9.4f} | {r['accuracy'][-1]:>11.4f}")

    # Save raw results
    save_data = {'default_acc': default_acc, **all_results}
    np.save('quant_results.npy', save_data)
    print("\nSaved per-step metrics to quant_results.npy")

    # --- Step 4: Plots ---
    steps = list(range(FLAGS.num_steps))
    colors = {'fp4': 'tab:blue', 'fp8': 'tab:orange', 'int8': 'tab:green', 'nf4': 'tab:red'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: L2 divergence
    ax = axes[0]
    for qt in quant_types:
        ax.plot(steps, all_results[qt]['l2_div'], label=qt, color=colors[qt])
    ax.set_xlabel('MCMC step')
    ax.set_ylabel('Normalized L2 divergence')
    ax.set_title('L2 divergence vs default trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax = axes[1]
    ax.plot(steps, default_acc, label='default', color='black', linewidth=2)
    for qt in quant_types:
        ax.plot(steps, all_results[qt]['accuracy'], label=qt, color=colors[qt])
    ax.set_xlabel('MCMC step')
    ax.set_ylabel('Energy score (-energy)')
    ax.set_title('Energy score over MCMC steps (higher = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quant_comparison.png', dpi=150)
    print("Saved comparison plot to quant_comparison.png")

    # Save final images (4×4 grid) for default and each quant type
    def save_grid(frame_cpu, name):
        out = frame_cpu.numpy().transpose(0, 2, 3, 1)
        out = out.reshape(4, 4, im_size, im_size, 3).transpose(0, 2, 1, 3, 4).reshape(4 * im_size, 4 * im_size, 3)
        imsave(f"quant_{name}.png", (out * 255).astype(np.uint8))

    save_grid(default_traj[-1], 'default')
    for qt in quant_types:
        save_grid(final_frames[qt], qt)
        print(f"Saved quant_{qt}.png")

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

    quantization_figure(model_list, select_idx)


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

