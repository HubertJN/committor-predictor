import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
import time

# ----------------- Raw HDF5 loading -----------------
def load_hdf5_raw(h5path, load_grids=True, indices=None):
    start = time.perf_counter()
    with h5py.File(h5path, "r") as f:
        total_saved = int(f['total_saved_grids'][()]) if 'total_saved_grids' in f else 0
        L = int(f['L'][()]) if 'L' in f else 0
        nbits = L * L
        nbytes = (nbits + 7) // 8

        if load_grids and 'grids' in f:
            if indices is None:
                raw_grids = f['grids'][()]
                grids = np.empty((total_saved, L, L), dtype=np.int8)
                for i in range(total_saved):
                    arr = np.frombuffer(raw_grids[i], dtype=np.uint8)
                    bits = np.unpackbits(arr, bitorder='little')[:nbits]
                    grids[i] = (bits.astype(np.int8) * 2 - 1).reshape(L, L)
            else:
                grids = np.empty((len(indices), L, L), dtype=np.int8)
                for i, idx in enumerate(indices):
                    arr = np.frombuffer(f['grids'][idx], dtype=np.uint8)
                    bits = np.unpackbits(arr, bitorder='little')[:nbits]
                    grids[i] = (bits.astype(np.int8) * 2 - 1).reshape(L, L)
        else:
            grids = np.empty((0, L, L), dtype=np.int8)

        header_keys = [key for key in f.keys() if key not in ('grids', 'attrs')]
        headers = {key: f[key][()] for key in header_keys}

        if indices is None:
            attrs = f['attrs'][()] if 'attrs' in f else np.empty((total_saved, 0))
        else:
            attrs = f['attrs'][indices] if 'attrs' in f else np.empty((len(indices), 0))

    end = time.perf_counter()
    if load_grids:
        print(f"Loaded {len(grids)} grids and attributes into memory.")
    else:
        print(f"Loaded headers and attributes for {len(attrs)} entries.")
    print(f"Elapsed time: {end - start:.2f} s")
    return grids, attrs, headers


# ----------------- CNN Dataset with Rotations + Reflections -----------------
class IsingDatasetCNN(Dataset):
    def __init__(self, grids, labels, device="cpu", augment=False):
        self.device = device

        grids = grids.to(device)
        labels = labels.to(device)

        augmented_grids = [grids]
        augmented_labels = [labels]

        if augment:
            # Add 90°, 180°, 270° rotations
            for k in [1, 2, 3]:
                rotated = torch.rot90(grids, k=k, dims=(2, 3))
                augmented_grids.append(rotated)
                augmented_labels.append(labels)

            # Collect reflections for each existing grid in augmented_grids
            new_grids = []
            new_labels = []
            for g, l in zip(augmented_grids, augmented_labels):
                # Horizontal flip (x-axis)
                new_grids.append(torch.flip(g, dims=[2]))
                new_labels.append(l)
                # Vertical flip (y-axis)
                new_grids.append(torch.flip(g, dims=[3]))
                new_labels.append(l)
                # Main diagonal (transpose x<->y)
                new_grids.append(g.transpose(2, 3))
                new_labels.append(l)
                # Anti-diagonal (flip + transpose)
                new_grids.append(torch.flip(g, dims=[2,3]).transpose(2, 3))
                new_labels.append(l)

            augmented_grids += new_grids
            augmented_labels += new_labels

        # Concatenate all augmented versions
        self.data = torch.cat(augmented_grids, dim=0)
        self.labels = torch.cat(augmented_labels, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def to_cnn_dataset(grids, attrs, device="cpu", augment=False):
    data_tensor = torch.tensor(grids, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(attrs[:, 2], dtype=torch.float32).unsqueeze(1)
    return IsingDatasetCNN(data_tensor, labels_tensor, device=device, augment=augment)

# ----------------- Uniform distribution filter -----------------
def uniform_filter(data, labels, num_bins=10, seed=42, total_samples=100000):
    np.random.seed(seed)
    bins = np.linspace(0, 1, num_bins + 1)
    
    # First, determine bin sizes
    bin_sizes = []
    for i in range(num_bins):
        bin_idx = np.where((labels >= bins[i]) & (labels < bins[i+1]))[0]
        bin_sizes.append(len(bin_idx))
        
    min_bin_size = min(bin_sizes)
    k = min(total_samples // num_bins, min_bin_size)
    
    num_to_take = [k] * num_bins
    total_taken = k * num_bins
    remaining = total_samples - total_taken
    
    # Remaining capacity per bin
    remaining_capacity = [bin_sizes[i] - k for i in range(num_bins)]
    
    # Sort bins by remaining capacity descending to prioritize balancing
    sorted_bins = sorted(range(num_bins), key=lambda i: remaining_capacity[i], reverse=True)
    
    for idx in sorted_bins:
        if remaining <= 0:
            break
        can_add = min(remaining, remaining_capacity[idx])
        num_to_take[idx] += can_add
        remaining -= can_add
    
    # Now, loop through bins and pull samples
    subset_indices = []
    for i in range(num_bins):
        bin_idx = np.where((labels >= bins[i]) & (labels < bins[i+1]))[0]
        selected = np.random.choice(bin_idx, size=num_to_take[i], replace=False)
        subset_indices.extend(selected)
    
    print(f"Selected {len(subset_indices)} samples as uniformly as possible across {num_bins} bins")
    return np.array(subset_indices)

# ----------------- Filter zero committor -----------------
def filter_zero_committor(grids, attrs):
    mask = attrs[:, 2] != 0
    grids_filtered = grids[mask]
    attrs_filtered = attrs[mask]
    print(f"Filtered {len(grids) - len(grids_filtered)} grids with committor=0")
    return grids_filtered, attrs_filtered

def data_to_dataloader(train_ds, valid_ds, test_ds, bs, model_type="cnn"):
    """
    Return DataLoaders for training and validation datasets.

    Args:
        train_ds, valid_ds: Dataset objects
        bs: batch size
        model_type: "cnn"
    """
    train_dl = TorchDataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = TorchDataLoader(valid_ds, batch_size=1)
    test_dl = TorchDataLoader(test_ds, batch_size=1)

    return train_dl, valid_dl, test_dl

def prepare_subset(h5path, uniform_bins=10, valid_size=0.2, test_size=0.2, seed=42, total_samples=None):
    """
    Load grids and attributes from HDF5 and select a uniform subset if total_samples is specified.

    Returns:
        grids_subset: np.ndarray
        attrs_subset: np.ndarray
        train_idx, valid_idx, test_idx: np.ndarray
    """
    print("Loading full dataset...")
    grids, attrs, _ = load_hdf5_raw(h5path)

    np.random.seed(seed)
    if total_samples is not None:
        print(f"Applying uniform filter to select {total_samples} samples")
        subset_indices = uniform_filter(grids, attrs[:,2], num_bins=uniform_bins, total_samples=total_samples)
        grids = grids[subset_indices]
        attrs = attrs[subset_indices]

    # print("Filtering 0 committor data")
    # grids, attrs = filter_zero_committor(grids, attrs)

    # print(f"Uniform subset size: {len(grids)}")

    # Split indices BEFORE augmentation
    num_samples = len(grids)
    indices = np.arange(num_samples)
    train_valid_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=valid_size / (1 - test_size), random_state=seed, shuffle=True
    )

    return grids, attrs, train_idx, valid_idx, test_idx

def prepare_datasets(grids, attrs, train_idx, valid_idx, test_idx, model_type="cnn", device="cpu", batch_size=64, augment=True):
    """
    Take grids and attributes with train/valid indices, and produce datasets and dataloaders.

    Returns:
        train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds
    """
    # Create train/validation datasets
    print(len(grids[train_idx]), len(grids[valid_idx]), len(grids[test_idx]))
    train_ds = to_cnn_dataset(grids[train_idx], attrs[train_idx],
                              device=device, augment=augment)
    valid_ds = to_cnn_dataset(grids[valid_idx], attrs[valid_idx],
                              device=device, augment=False)
    test_ds = to_cnn_dataset(grids[test_idx], attrs[test_idx],
                              device=device, augment=False)

    print(f"Train size: {len(train_ds)}, Validation size: {len(valid_ds)}, Test size: {len(test_ds)}")

    # Create DataLoaders
    train_dl, valid_dl, test_dl = data_to_dataloader(train_ds, valid_ds, test_ds, batch_size,
                                            model_type=model_type.lower())
    print(f"DataLoaders created: train_batches={len(train_dl)}, valid_batches={len(valid_dl)}, test_batches={len(test_dl)}")

    return train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds
