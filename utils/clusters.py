from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def largest_cluster_size_up(grid_2d: np.ndarray) -> int:
    """Largest nearest-neighbour geometric cluster of up spins."""

    labeled_array, num_features = label(np.asarray(grid_2d) > 0)
    if num_features == 0:
        return 0

    sizes = np.bincount(labeled_array.ravel())
    if sizes.size <= 1:
        return 0
    return int(sizes[1:].max())


def largest_cluster_sizes_up(frames: np.ndarray) -> np.ndarray:
    """Batch wrapper for largest_cluster_size_up."""

    frames = np.asarray(frames)
    out = np.empty(frames.shape[0], dtype=np.float64)
    for i in range(frames.shape[0]):
        out[i] = largest_cluster_size_up(frames[i])
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[x] != x:
            nxt = int(self.parent[x])
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _child_seed(seed: int, beta: float, h: float | None, index: int) -> int:
    h_part = 0 if h is None else int(round(float(h) * 1_000_000))
    seq = np.random.SeedSequence(
        [
            int(seed),
            int(round(float(beta) * 1_000_000)),
            h_part,
            int(index),
        ]
    )
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def fk_largest_cluster_size_up(
    grid_2d: np.ndarray,
    beta: float,
    seed: int,
) -> int:
    """Largest single-draw Fortuin-Kasteleyn cluster of up spins.

    Bonds are sampled only between nearest-neighbour up-spin pairs, matching the
    nucleating-droplet convention used by the geometric LCS baseline. The
    neighbour convention is intentionally the same open-boundary convention as
    scipy.ndimage.label in largest_cluster_size_up.
    """

    grid = np.asarray(grid_2d)
    up = grid > 0
    n_up = int(np.count_nonzero(up))
    if n_up == 0:
        return 0
    if n_up == 1:
        return 1

    p_bond = 1.0 - np.exp(-2.0 * float(beta))
    p_bond = float(np.clip(p_bond, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))

    labels = -np.ones(up.shape, dtype=np.int64)
    labels[up] = np.arange(n_up, dtype=np.int64)
    uf = _UnionFind(n_up)

    vertical = up[:-1, :] & up[1:, :]
    if np.any(vertical):
        a_rows, a_cols = np.nonzero(vertical)
        keep = rng.random(a_rows.size) < p_bond
        for r, c in zip(a_rows[keep], a_cols[keep]):
            uf.union(int(labels[r, c]), int(labels[r + 1, c]))

    horizontal = up[:, :-1] & up[:, 1:]
    if np.any(horizontal):
        a_rows, a_cols = np.nonzero(horizontal)
        keep = rng.random(a_rows.size) < p_bond
        for r, c in zip(a_rows[keep], a_cols[keep]):
            uf.union(int(labels[r, c]), int(labels[r, c + 1]))

    roots = np.fromiter((uf.find(i) for i in range(n_up)), dtype=np.int64, count=n_up)
    return int(np.bincount(roots).max())


def fk_largest_cluster_sizes_up(
    frames: np.ndarray,
    beta: float,
    seed: int,
    h: float | None = None,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Batch wrapper for seeded single-draw up-spin FK clusters."""

    frames = np.asarray(frames)
    if indices is None:
        indices = np.arange(frames.shape[0], dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape[0] != frames.shape[0]:
            raise ValueError("indices must have the same length as frames")

    out = np.empty(frames.shape[0], dtype=np.float64)
    for i in range(frames.shape[0]):
        out[i] = fk_largest_cluster_size_up(
            frames[i],
            beta=float(beta),
            seed=_child_seed(int(seed), float(beta), h, int(indices[i])),
        )
    return out
