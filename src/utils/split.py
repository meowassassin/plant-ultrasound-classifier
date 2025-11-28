# src/utils/split.py
from typing import List, Tuple
import numpy as np


def make_lopo_splits_from_plant_ids(
    plant_ids: List[int],
) -> List[Tuple[List[int], List[int]]]:
    """Make Leave-One-Plant-Out splits from plant_ids.

    Args:
        plant_ids: length N list, plant_ids[i] is plant id of sample i.

    Returns:
        List of (train_indices, test_indices),
        indices are positions (0..N-1).
    """
    plant_ids = list(plant_ids)
    n = len(plant_ids)
    unique_ids = sorted(set(plant_ids))

    # ignore invalid id -1 for LOPO, will keep them in train always
    valid_ids = [pid for pid in unique_ids if pid >= 0]

    splits: List[Tuple[List[int], List[int]]] = []

    if not valid_ids:
        # fallback: single 80/20 split
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        n_test = max(1, int(0.2 * n))
        test_idx = idxs[:n_test].tolist()
        train_idx = idxs[n_test:].tolist()
        splits.append((train_idx, test_idx))
        return splits

    for pid in valid_ids:
        test_idx = [i for i in range(n) if plant_ids[i] == pid]
        if not test_idx:
            continue
        train_idx = [i for i in range(n) if i not in test_idx]
        if not train_idx:
            continue
        splits.append((train_idx, test_idx))

    return splits
