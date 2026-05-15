from __future__ import annotations

import numpy as np

from utils.clusters import (
    fk_largest_cluster_size_up,
    fk_largest_cluster_sizes_up,
    largest_cluster_size_up,
)


def main() -> None:
    empty = -np.ones((4, 4), dtype=np.int8)
    assert largest_cluster_size_up(empty) == 0
    assert fk_largest_cluster_size_up(empty, beta=0.5, seed=1) == 0

    one = empty.copy()
    one[1, 1] = 1
    assert largest_cluster_size_up(one) == 1
    assert fk_largest_cluster_size_up(one, beta=0.5, seed=1) == 1

    block = empty.copy()
    block[1:3, 1:3] = 1
    assert largest_cluster_size_up(block) == 4

    fk_a = fk_largest_cluster_size_up(block, beta=0.5, seed=123)
    fk_b = fk_largest_cluster_size_up(block, beta=0.5, seed=123)
    assert fk_a == fk_b
    assert 1 <= fk_a <= 4

    low_beta = fk_largest_cluster_size_up(block, beta=0.0, seed=123)
    assert low_beta == 1

    high_beta = fk_largest_cluster_size_up(block, beta=100.0, seed=123)
    assert high_beta == 4

    frames = np.stack([empty, one, block], axis=0)
    idx = np.array([5, 6, 7], dtype=np.int64)

    avg_a = fk_largest_cluster_sizes_up(frames, beta=0.5, seed=2024, h=0.3, indices=idx, n_draws=32)
    avg_b = fk_largest_cluster_sizes_up(frames, beta=0.5, seed=2024, h=0.3, indices=idx, n_draws=32)
    assert np.allclose(avg_a, avg_b)
    assert avg_a.shape == (3,)
    assert avg_a[0] == 0.0
    assert avg_a[1] == 1.0
    assert 1.0 <= avg_a[2] <= 4.0

    avg_low = fk_largest_cluster_sizes_up(np.stack([block]), beta=0.0, seed=7, h=0.3, indices=np.array([0]), n_draws=128)[0]
    assert np.isclose(avg_low, 1.0)
    avg_high = fk_largest_cluster_sizes_up(np.stack([block]), beta=100.0, seed=7, h=0.3, indices=np.array([0]), n_draws=128)[0]
    assert np.isclose(avg_high, 4.0)

    try:
        fk_largest_cluster_sizes_up(frames, beta=0.5, seed=1, n_draws=0)
        raise AssertionError("Expected ValueError for n_draws=0")
    except ValueError:
        pass

    print("cluster tests passed")


if __name__ == "__main__":
    main()
