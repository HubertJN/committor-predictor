from __future__ import annotations

import numpy as np

from utils.clusters import (
    fk_largest_cluster_size_up,
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

    print("cluster tests passed")


if __name__ == "__main__":
    main()
