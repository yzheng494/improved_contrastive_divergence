import numpy as np

data = np.load('cross_step_quant_results_k10/quant_results.npy', allow_pickle=True).item()

# Collect (fid_list, precision_list) per setting
settings = {}

# Default has top-level keys
settings['default'] = {
    'fid': data['default_fid'],
    'precision': data['default_precision'],
}

# All other settings are nested dicts
for key in ['fp4', 'fp8', 'int8', 'nf4', 'cross_step',
            'cross_step_fp4', 'cross_step_fp8', 'cross_step_int8', 'cross_step_nf4']:
    settings[key] = {
        'fid': data[key]['fid'],
        'precision': data[key]['precision'],
    }

print(f"{'Setting':<20} {'Best FID':>12} {'Step@FID':>10} {'Best Prec':>12} {'Step@Prec':>12}")
print("-" * 70)

global_best_fid = (float('inf'), None, None)
global_best_prec = (-float('inf'), None, None)

for name, metrics in settings.items():
    # Filter out None values, keeping track of original indices
    fid_pairs = [(i, v) for i, v in enumerate(metrics['fid']) if v is not None]
    prec_pairs = [(i, v) for i, v in enumerate(metrics['precision']) if v is not None]

    best_fid_idx, best_fid_val = min(fid_pairs, key=lambda x: x[1])
    best_prec_idx, best_prec_val = max(prec_pairs, key=lambda x: x[1])

    print(f"{name:<20} {best_fid_val:>12.4f} {best_fid_idx:>10} {best_prec_val:>12.4f} {best_prec_idx:>12}")

    if best_fid_val < global_best_fid[0]:
        global_best_fid = (best_fid_val, best_fid_idx, name)
    if best_prec_val > global_best_prec[0]:
        global_best_prec = (best_prec_val, best_prec_idx, name)

print("-" * 70)
print(f"\nGlobal best FID:       {global_best_fid[0]:.4f}  (setting={global_best_fid[2]}, step={global_best_fid[1]})")
print(f"Global best Precision: {global_best_prec[0]:.4f}  (setting={global_best_prec[2]}, step={global_best_prec[1]})")
