# src/datasets/plantsounds.py
import os
from typing import List, Tuple

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..utils.audio import load_waveform_1d


@dataclass
class SampleMeta:
    idx: int
    rel_path: str
    species: str   # 'tomato', 'tobacco', 'none'
    stress: str    # 'dry', 'cut', 'none'
    domain: str    # 'box', 'greenhouse'
    plant_id: int  # parsed from filename if available
    group: str     # folder name (e.g. 'Tomato Dry')


def parse_plant_id(filename: str) -> int:
    """Parse plant id from a filename like 'id_0_sound_1.wav'.

    If parse fails, returns -1.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    try:
        for i, p in enumerate(parts):
            if p == 'id' and i + 1 < len(parts):
                return int(parts[i + 1])
            if p.startswith('id') and len(p) > 2:
                # e.g. id0
                return int(p[2:])
    except ValueError:
        pass
    return -1


def scan_plantsounds(root: str) -> List[SampleMeta]:
    """Scan the PlantSounds directory and build metadata list.

    Expected subdirectories under root:
      - 'Tomato Dry'
      - 'Tomato Cut'
      - 'Tobacco Dry'
      - 'Tobacco Cut'
      - 'Empty Pot'
      - 'Greenhouse Noises'
    """
    metas: List[SampleMeta] = []
    idx = 0

    if not os.path.isdir(root):
        raise FileNotFoundError(f"PlantSounds root not found: {root}")

    for group in sorted(os.listdir(root)):
        group_path = os.path.join(root, group)
        if not os.path.isdir(group_path):
            continue

        # Determine species, stress, domain from folder name
        folder_lower = group.lower()
        if 'tomato' in folder_lower:
            species = 'tomato'
        elif 'tobacco' in folder_lower:
            species = 'tobacco'
        else:
            species = 'none'

        if 'dry' in folder_lower:
            stress = 'dry'
        elif 'cut' in folder_lower:
            stress = 'cut'
        else:
            stress = 'none'

        if 'greenhouse' in folder_lower:
            domain = 'greenhouse'
        else:
            # all others (Tomato*, Tobacco*, Empty Pot) are acoustic box recordings
            domain = 'box'

        for fname in sorted(os.listdir(group_path)):
            if not fname.lower().endswith('.wav'):
                continue
            rel_path = os.path.join(group, fname)
            plant_id = parse_plant_id(fname)
            metas.append(
                SampleMeta(
                    idx=idx,
                    rel_path=rel_path,
                    species=species,
                    stress=stress,
                    domain=domain,
                    plant_id=plant_id,
                    group=group,
                )
            )
            idx += 1

    return metas


def filter_for_task(meta: SampleMeta, task_name: str) -> bool:
    """Return True if this sample should be used for the given task."""
    t = task_name

    # Task1 variants: Dry/Cut & species comparisons in acoustic box
    if t == 'task1_tomato_dry_vs_cut':
        return meta.domain == 'box' and meta.species == 'tomato' and meta.stress in {'dry', 'cut'}
    if t == 'task1_tobacco_dry_vs_cut':
        return meta.domain == 'box' and meta.species == 'tobacco' and meta.stress in {'dry', 'cut'}
    if t == 'task1_dry_tomato_vs_tobacco':
        return meta.domain == 'box' and meta.stress == 'dry' and meta.species in {'tomato', 'tobacco'}
    if t == 'task1_cut_tomato_vs_tobacco':
        return meta.domain == 'box' and meta.stress == 'cut' and meta.species in {'tomato', 'tobacco'}

    # Task2: plant vs empty pot (box noise)
    if t == 'task2_plant_vs_empty':
        return meta.domain == 'box' and (
            (meta.species in {'tomato', 'tobacco'} and meta.stress in {'dry', 'cut'})
            or meta.group.lower().startswith('empty pot')
        )

    # Task3: tomato vs greenhouse noise
    if t == 'task3_tomato_vs_greenhouse':
        return (
            (meta.group.lower().startswith('tomato') and meta.stress == 'dry')
            or meta.group.lower().startswith('greenhouse noise')
        )

    raise ValueError(f"Unknown task_name: {task_name}")


def make_label(meta: SampleMeta, task_name: str) -> int:
    """Map SampleMeta to an integer label for a given task."""
    t = task_name

    if t == 'task1_tomato_dry_vs_cut':
        # dry=1, cut=0
        return 1 if meta.stress == 'dry' else 0

    if t == 'task1_tobacco_dry_vs_cut':
        return 1 if meta.stress == 'dry' else 0

    if t == 'task1_dry_tomato_vs_tobacco':
        # tomato=0, tobacco=1
        return 0 if meta.species == 'tomato' else 1

    if t == 'task1_cut_tomato_vs_tobacco':
        return 0 if meta.species == 'tomato' else 1

    if t == 'task2_plant_vs_empty':
        # plant=1, empty pot=0
        if meta.group.lower().startswith('empty pot'):
            return 0
        return 1

    if t == 'task3_tomato_vs_greenhouse':
        # tomato dry=1, greenhouse noise=0
        if meta.group.lower().startswith('greenhouse noise'):
            return 0
        return 1

    raise ValueError(f"Unknown task_name: {task_name}")


def make_lopo_splits(
    metas: List[SampleMeta],
    task_name: str,
) -> List[Tuple[List[int], List[int]]]:
    """Make Leave-One-Plant-Out splits for a given task.

    Returns:
        list of (train_indices, test_indices),
        where indices are positions in `metas`.
    """
    # Filter metas for this task first
    used_indices = [i for i, m in enumerate(metas) if filter_for_task(m, task_name)]
    used_metas = [metas[i] for i in used_indices]

    # plant_ids for this task
    plant_ids = sorted({m.plant_id for m in used_metas})
    # If plant_id is -1 (parse failed), treat them as a single pseudo-plant
    # and keep them always in train (never as a held-out test plant)
    plant_ids_for_cv = [pid for pid in plant_ids if pid >= 0]

    splits: List[Tuple[List[int], List[int]]] = []

    if not plant_ids_for_cv:
        # Fallback: just one split train/test 80/20 random
        n = len(used_indices)
        n_test = max(1, int(0.2 * n))
        test_idx = used_indices[:n_test]
        train_idx = used_indices[n_test:]
        splits.append((train_idx, test_idx))
        return splits

    for pid in plant_ids_for_cv:
        test_global_idx = [i for i in used_indices if metas[i].plant_id == pid]
        if not test_global_idx:
            continue
        train_global_idx = [i for i in used_indices if i not in test_global_idx]
        if not train_global_idx:
            continue
        splits.append((train_global_idx, test_global_idx))

    return splits


class PlantSoundsDataset(Dataset):
    """Dataset that loads 1D waveforms for a specific task.

    It uses the 1D waveform prepared for Conv1d (shape [1, L]).
    """

    def __init__(
        self,
        root: str,
        metas: List[SampleMeta],
        indices: List[int],
        task_name: str,
        target_sr: int = 500000,
        target_len: int = 1000,
        apply_highpass: bool = True,
    ):
        self.root = root
        self.metas = metas
        self.indices = indices
        self.task_name = task_name
        self.target_sr = target_sr
        self.target_len = target_len
        self.apply_highpass = apply_highpass

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        meta_idx = self.indices[idx]
        meta = self.metas[meta_idx]
        full_path = os.path.join(self.root, meta.rel_path)

        x = load_waveform_1d(
            full_path,
            target_sr=self.target_sr,
            target_len=self.target_len,
            apply_highpass=self.apply_highpass,
        )  # [1, L]

        y = make_label(meta, self.task_name)
        y_t = torch.tensor(y, dtype=torch.long)
        return x, y_t
