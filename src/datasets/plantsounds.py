# src/datasets/plantsounds.py
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

import numpy as np
from collections import Counter

from src.utils.audio import load_waveform_1d

# Minimum number of plant samples to use in LOPO
# (if only one event exists with this plant_id in the task, don't use it for LOPO)
MIN_PLANT_SAMPLES = 1


@dataclass
class SampleMeta:
    idx: int
    rel_path: str
    species: str   # 'tomato', 'tobacco', 'none'
    stress: str    # 'dry', 'cut', 'none'
    domain: str    # 'box', 'greenhouse'
    plant_id: int  # parsed from filename
    group: str     # folder name (e.g. 'Tomato Dry')


def parse_plant_id(filename: str) -> int:
    """Parse plant id from filename like 'id_0_sound_1.wav'.

    If parse fails, returns -1.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    try:
        for i, p in enumerate(parts):
            if p == 'id' and i + 1 < len(parts):
                return int(parts[i + 1])
            if p.startswith('id') and len(p) > 2:
                return int(p[2:])
    except ValueError:
        pass
    return -1


def scan_plantsounds(root: str) -> List[SampleMeta]:
    """Scan data/raw/PlantSounds and build SampleMeta list."""
    metas: List[SampleMeta] = []
    idx = 0

    if not os.path.isdir(root):
        raise FileNotFoundError(f"PlantSounds root not found: {root}")

    for group in sorted(os.listdir(root)):
        group_path = os.path.join(root, group)
        if not os.path.isdir(group_path):
            continue

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
    """Return True if this sample is used for given task."""
    t = task_name

    # Task1 variants (acoustic box)
    if t == 'task1_tomato_dry_vs_cut':
        return meta.domain == 'box' and meta.species == 'tomato' and meta.stress in {'dry', 'cut'}
    if t == 'task1_tobacco_dry_vs_cut':
        return meta.domain == 'box' and meta.species == 'tobacco' and meta.stress in {'dry', 'cut'}
    if t == 'task1_dry_tomato_vs_tobacco':
        return meta.domain == 'box' and meta.stress == 'dry' and meta.species in {'tomato', 'tobacco'}
    if t == 'task1_cut_tomato_vs_tobacco':
        return meta.domain == 'box' and meta.stress == 'cut' and meta.species in {'tomato', 'tobacco'}

    # Task2: plant vs Empty Pot
    if t == 'task2_plant_vs_empty':
        return meta.domain == 'box' and (
            (meta.species in {'tomato', 'tobacco'} and meta.stress in {'dry', 'cut'})
            or meta.group.lower().startswith('empty pot')
        )

    # Task3: Tomato Dry vs Greenhouse noise
    if t == 'task3_tomato_vs_greenhouse':
        return (
            (meta.group.lower().startswith('tomato') and meta.stress == 'dry')
            or meta.group.lower().startswith('greenhouse noise')
        )

    # Task5: All tomato<->tobacco dry/cut (split by species in train/test)
    if t == 'task5_tomato2tobacco_dry_vs_cut':
        return (
            meta.domain == 'box'
            and meta.species in {'tomato', 'tobacco'}
            and meta.stress in {'dry', 'cut'}
        )
    
    raise ValueError(f"Unknown task_name: {task_name}")
    
    

def make_label(meta: SampleMeta, task_name: str) -> int:
    """Map SampleMeta to integer label for a given task."""
    t = task_name

    # Task1
    if t in {'task1_tomato_dry_vs_cut', 'task1_tobacco_dry_vs_cut'}:
        # dry=1, cut=0
        return 1 if meta.stress == 'dry' else 0

    if t in {'task1_dry_tomato_vs_tobacco', 'task1_cut_tomato_vs_tobacco'}:
        # tomato=0, tobacco=1
        return 0 if meta.species == 'tomato' else 1

    # Task2: plant vs Empty Pot
    if t == 'task2_plant_vs_empty':
        if meta.group.lower().startswith('empty pot'):
            return 0
        return 1

    # Task3: tomato dry vs greenhouse noise
    if t == 'task3_tomato_vs_greenhouse':
        if meta.group.lower().startswith('greenhouse noise'):
            return 0
        return 1

    # Task5: dry(1) vs cut(0)
    if t == 'task5_tomato2tobacco_dry_vs_cut':
        return 1 if meta.stress == 'dry' else 0
    
    raise ValueError(f"Unknown task_name: {task_name}")


def make_task_lopo_splits(
    metas: List[SampleMeta],
    task_name: str,
) -> List[Tuple[List[int], List[int]]]:
    """
    Paper-style LOPO (Leave-One-Plant-Out) implementation.

    Common:
      - First select only samples used for the given task to create used_indices / used_metas.
      - used_indices[j]: index in the full metas list (global)
      - used_metas[j]: SampleMeta (local)

    Task1 (plant vs plant):
      - For all used plant_id (>=0),
        use all sounds from one plant as test and the rest as train (LOPO).

    Task2/Task3 (plant vs noise: Empty Pot / Greenhouse):
      - plant (label=1) uses plant_id-based LOPO.
      - noise (label=0) is divided into K groups (same as number of plant folds),
        fold i uses plant_id=unique_pids[i] and noise_group[i] as test.
      - Each fold's test is designed to include both plant + noise classes.
    """
    # 1) Use only samples for this task
    used_indices = [i for i, m in enumerate(metas) if filter_for_task(m, task_name)]
    used_metas = [metas[i] for i in used_indices]

    if not used_metas:
        raise RuntimeError(f"No samples found for task: {task_name}")

    # Check if this is Task2/3 (plant vs noise series)
    is_noise_task = task_name in {"task2_plant_vs_empty", "task3_tomato_vs_greenhouse"}

    # --------------------------
    # A. plant vs noise (Task2/3)
    # --------------------------
    if is_noise_task:
        # Calculate labels based on local index
        local_labels = [make_label(m, task_name) for m in used_metas]
        plant_locals = [li for li, y in enumerate(local_labels) if y == 1]
        noise_locals = [li for li, y in enumerate(local_labels) if y == 0]

        if not plant_locals or not noise_locals:
            raise RuntimeError(
                f"Task {task_name}: Insufficient plant or noise classes."
            )

        # Convert to global index (store plant / noise separately)
        plant_globals = [used_indices[li] for li in plant_locals]
        noise_globals = [used_indices[li] for li in noise_locals]

        # Collect plant_id for plant-side LOPO
        plant_metas = [metas[g] for g in plant_globals]
        plant_ids = [m.plant_id for m in plant_metas]

        # Use only plant_id >= 0
        unique_pids = sorted({pid for pid in plant_ids if pid is not None and pid >= 0})
        if not unique_pids:
            # If no plant_id info, statistically meaningful LOPO not possible → 80/20 random split fallback
            all_locals = np.arange(len(used_indices))
            rng = np.random.RandomState(42)
            rng.shuffle(all_locals)
            n_test = max(1, int(0.2 * len(all_locals)))
            test_local = all_locals[:n_test].tolist()
            train_local = all_locals[n_test:].tolist()
            train_global = [used_indices[i] for i in train_local]
            test_global = [used_indices[i] for i in test_local]
            return [(train_global, test_global)]

        K = len(unique_pids)  # Number of folds = number of plant individuals

        # --- Split noise into K groups (paper: noise also divided into groups for CV) ---
        rng = np.random.RandomState(42)
        noise_arr = np.array(noise_globals)
        rng.shuffle(noise_arr)

        noise_groups: List[List[int]] = [[] for _ in range(K)]
        for j, g in enumerate(noise_arr):
            noise_groups[j % K].append(int(g))

        splits: List[Tuple[List[int], List[int]]] = []

        # fold i: use plant_id = unique_pids[i] and noise_group[i] as test
        for i, pid in enumerate(unique_pids):
            # Global indices of plant samples with this plant_id
            test_pl_globals = [
                g for g in plant_globals
                if metas[g].plant_id == pid
            ]
            train_pl_globals = [
                g for g in plant_globals
                if metas[g].plant_id != pid
            ]

            noise_test = noise_groups[i]
            noise_train = []
            for j, grp in enumerate(noise_groups):
                if j != i:
                    noise_train.extend(grp)

            # If test doesn't have both classes, don't use as fold
            if not test_pl_globals or not noise_test:
                continue

            train_global = train_pl_globals + noise_train
            test_global = test_pl_globals + noise_test

            if not train_global or not test_global:
                continue

            splits.append((train_global, test_global))

        # If no folds were created above → 80/20 random split fallback
        if not splits:
            all_locals = np.arange(len(used_indices))
            rng = np.random.RandomState(42)
            rng.shuffle(all_locals)
            n_test = max(1, int(0.2 * len(all_locals)))
            test_local = all_locals[:n_test].tolist()
            train_local = all_locals[n_test:].tolist()
            train_global = [used_indices[i] for i in train_local]
            test_global = [used_indices[i] for i in test_local]
            splits.append((train_global, test_global))

        return splits

    # --------------------------
    # B. plant vs plant (Task1 series)
    # --------------------------
    # As stated in paper:
    #   "leave all the emitted sounds of one plant out for cross validation"
    plant_ids_task = [m.plant_id for m in used_metas]
    unique_pids = sorted({pid for pid in plant_ids_task if pid is not None and pid >= 0})
    splits: List[Tuple[List[int], List[int]]] = []

    for pid in unique_pids:
        # Samples with this plant_id (local index)
        test_locals = [
            li for li, m in enumerate(used_metas)
            if m.plant_id == pid
        ]
        train_locals = [
            li for li, m in enumerate(used_metas)
            if m.plant_id != pid
        ]

        if not test_locals or not train_locals:
            continue

        train_global = [used_indices[li] for li in train_locals]
        test_global = [used_indices[li] for li in test_locals]
        splits.append((train_global, test_global))

    # If no plant_id info and LOPO not possible → 80/20 random split fallback
    if not splits:
        all_locals = np.arange(len(used_indices))
        rng = np.random.RandomState(42)
        rng.shuffle(all_locals)
        n_test = max(1, int(0.2 * len(all_locals)))
        test_local = all_locals[:n_test].tolist()
        train_local = all_locals[n_test:].tolist()
        train_global = [used_indices[i] for i in train_local]
        test_global = [used_indices[i] for i in test_local]
        splits.append((train_global, test_global))

    return splits



class PlantSoundsDataset(Dataset):
    """1D waveform dataset for a given task."""

    def __init__(
        self,
        root: str,
        metas: List[SampleMeta],
        indices: List[int],
        task_name: str,
        target_sr: int = 500000,
        target_len: int = 1000,  # 2ms @ 500kHz
        apply_highpass: bool = True,
        return_domain: bool = False,
    ):
        self.root = root
        self.metas = metas
        self.indices = indices
        self.task_name = task_name
        self.target_sr = target_sr
        self.target_len = target_len
        self.apply_highpass = apply_highpass
        self.return_domain = return_domain
        

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
        
        if not self.return_domain:
            return x, y_t
        
        d = 0 if meta.domain == 'box' else 1
        d_t = torch.tensor(d, dtype=torch.long)
        return x, y_t, d_t
