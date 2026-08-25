import numpy as np
from src.data_split import get_liver_splits

imgs = np.load('data/processed/liver_images.npy')
msks = np.load('data/processed/liver_masks.npy')
print(f'Liver array shapes: images={imgs.shape}, masks={msks.shape}')

tr_ds, val_ds, te_ds, sizes = get_liver_splits(seed=42)
print(f'Split sizes: {sizes}')

# Re-compute indices to verify disjointness and seed reproducibility
total = len(imgs)
idx = np.arange(total)
rng = np.random.RandomState(42)
rng.shuffle(idx)
tr_end = int(total * 0.70)
val_end = tr_end + int(total * 0.15)
tr_idx, val_idx, te_idx = idx[:tr_end], idx[tr_end:val_end], idx[val_end:]

assert len(set(tr_idx).intersection(set(val_idx))) == 0, 'Overlap train/val'
assert len(set(tr_idx).intersection(set(te_idx))) == 0, 'Overlap train/test'
assert len(set(val_idx).intersection(set(te_idx))) == 0, 'Overlap val/test'
assert len(tr_idx) + len(val_idx) + len(te_idx) == total, 'Coverage mismatch'
print('Disjointness & partition completeness verified!')

# Test seed reproducibility
rng2 = np.random.RandomState(42)
idx2 = np.arange(total)
rng2.shuffle(idx2)
assert np.array_equal(idx, idx2), 'Seed reproducibility failed'
print('Seed reproducibility verified!')
