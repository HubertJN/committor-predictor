from __future__ import annotations

import numpy as np
from numba import njit
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


@njit(cache=True, inline="always")
def _uf_find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != x:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True, inline="always")
def _uf_union(parent: np.ndarray, size: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        tmp = ra
        ra = rb
        rb = tmp
    parent[rb] = ra
    size[ra] += size[rb]


@njit(cache=True)
def _fk_largest_from_edges(
    n_up: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    keep: np.ndarray,
) -> int:
    parent = np.arange(n_up, dtype=np.int64)
    size = np.ones(n_up, dtype=np.int64)

    for i in range(edge_u.shape[0]):
        if keep[i]:
            _uf_union(parent, size, int(edge_u[i]), int(edge_v[i]))

    # Count component sizes after full compression.
    counts = np.zeros(n_up, dtype=np.int64)
    max_count = 1
    for i in range(n_up):
        r = _uf_find(parent, i)
        counts[r] += 1
        if counts[r] > max_count:
            max_count = counts[r]
    return int(max_count)


def _child_seed(seed: int, beta: float, h: float | None, index: int, draw_index: int = 0) -> int:
    h_part = 0 if h is None else int(round(float(h) * 1_000_000))
    seq = np.random.SeedSequence(
        [
            int(seed),
            int(round(float(beta) * 1_000_000)),
            h_part,
            int(index),
            int(draw_index),
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

    vertical = up[:-1, :] & up[1:, :]
    v_u = np.empty(0, dtype=np.int64)
    v_v = np.empty(0, dtype=np.int64)
    keep_v = np.empty(0, dtype=np.bool_)
    if np.any(vertical):
        a_rows, a_cols = np.nonzero(vertical)
        keep_v = rng.random(a_rows.size) < p_bond
        v_u = labels[a_rows, a_cols].astype(np.int64, copy=False)
        v_v = labels[a_rows + 1, a_cols].astype(np.int64, copy=False)

    horizontal = up[:, :-1] & up[:, 1:]
    h_u = np.empty(0, dtype=np.int64)
    h_v = np.empty(0, dtype=np.int64)
    keep_h = np.empty(0, dtype=np.bool_)
    if np.any(horizontal):
        a_rows, a_cols = np.nonzero(horizontal)
        keep_h = rng.random(a_rows.size) < p_bond
        h_u = labels[a_rows, a_cols].astype(np.int64, copy=False)
        h_v = labels[a_rows, a_cols + 1].astype(np.int64, copy=False)

    edge_u = np.concatenate((v_u, h_u))
    edge_v = np.concatenate((v_v, h_v))
    keep = np.concatenate((keep_v, keep_h))
    return int(_fk_largest_from_edges(n_up, edge_u, edge_v, keep))


def fk_largest_cluster_sizes_up(
    frames: np.ndarray,
    beta: float,
    seed: int,
    h: float | None = None,
    indices: np.ndarray | None = None,
    n_draws: int = 1,
) -> np.ndarray:
    """Batch wrapper for seeded up-spin FK clusters averaged over draws."""

    frames = np.asarray(frames)
    if indices is None:
        indices = np.arange(frames.shape[0], dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape[0] != frames.shape[0]:
            raise ValueError("indices must have the same length as frames")
    n_draws = int(n_draws)
    if n_draws <= 0:
        raise ValueError("n_draws must be >= 1")

    out = np.empty(frames.shape[0], dtype=np.float64)
    for i in range(frames.shape[0]):
        draw_sizes = np.empty(n_draws, dtype=np.float64)
        for draw_idx in range(n_draws):
            draw_sizes[draw_idx] = fk_largest_cluster_size_up(
                frames[i],
                beta=float(beta),
                seed=_child_seed(int(seed), float(beta), h, int(indices[i]), draw_index=draw_idx),
            )
        out[i] = float(np.mean(draw_sizes))
    return out
