from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .utils import seed_everything


@dataclass
class TaskData:
    A_train: DataLoader
    B_train: DataLoader
    A_test: DataLoader
    B_test: DataLoader
    n_classes: int
    image_size: int


def _image_key(row: dict) -> str:
    for key in ("img", "image"):
        if key in row:
            return key
    raise KeyError(f"could not find image key in row keys={list(row)}")


def _label_key(row: dict) -> str:
    for key in ("label", "fine_label", "class"):
        if key in row:
            return key
    raise KeyError(f"could not find label key in row keys={list(row)}")


class HFClassTask(Dataset):
    def __init__(
        self,
        hf_split,
        classes: Sequence[int],
        per_class: int,
        image_size: int,
        normalize_mean: tuple[float, float, float],
        normalize_std: tuple[float, float, float],
        train: bool,
    ):
        self.classes = tuple(int(c) for c in classes)
        self.label_map = {c: i for i, c in enumerate(self.classes)}
        counts = {c: 0 for c in self.classes}
        rows = []
        first = hf_split[0]
        self.image_key = _image_key(first)
        self.label_key = _label_key(first)
        for row in hf_split:
            y = int(row[self.label_key])
            if y in counts and counts[y] < per_class:
                rows.append(row)
                counts[y] += 1
            if all(v >= per_class for v in counts.values()):
                break
        missing = {c: per_class - n for c, n in counts.items() if n < per_class}
        if missing:
            print(f"WARNING: insufficient examples for some classes: {missing}; using available examples")
        self.rows = rows
        train_ops = [
            v2.ToImage(),
            v2.RandomCrop(image_size, padding=max(4, image_size // 8)),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(normalize_mean, normalize_std),
        ]
        eval_ops = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(normalize_mean, normalize_std),
        ]
        self.transform = v2.Compose(train_ops if train else eval_ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        image = row[self.image_key]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        x = self.transform(image)
        y = self.label_map[int(row[self.label_key])]
        return x, y


def make_task_data(cfg, dataset: str, seed: int) -> TaskData:
    seed_everything(seed)
    if dataset == "split_cifar10":
        hf = load_dataset(cfg.cifar10_dataset_id)
        train_split, test_split = hf["train"], hf["test"]
        task_a, task_b = cfg.cifar10_task_a, cfg.cifar10_task_b
        train_per, test_per = cfg.cifar_train_per_class, cfg.cifar_test_per_class
        image_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == "split_tinyimagenet":
        hf = load_dataset(cfg.tinyimagenet_dataset_id)
        valid_name = "valid" if "valid" in hf else "validation"
        train_split, test_split = hf["train"], hf[valid_name]
        task_a, task_b = cfg.tiny_task_a, cfg.tiny_task_b
        train_per, test_per = cfg.tiny_train_per_class, cfg.tiny_test_per_class
        image_size = 64
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    else:
        raise ValueError(dataset)
    n_classes = len(task_a)
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    g = torch.Generator().manual_seed(int(seed))
    A_train = HFClassTask(train_split, task_a, train_per, image_size, mean, std, True)
    B_train = HFClassTask(train_split, task_b, train_per, image_size, mean, std, True)
    A_test = HFClassTask(test_split, task_a, test_per, image_size, mean, std, False)
    B_test = HFClassTask(test_split, task_b, test_per, image_size, mean, std, False)
    return TaskData(
        DataLoader(A_train, shuffle=True, generator=g, **kw),
        DataLoader(B_train, shuffle=True, generator=torch.Generator().manual_seed(int(seed) + 17), **kw),
        DataLoader(A_test, shuffle=False, **kw),
        DataLoader(B_test, shuffle=False, **kw),
        n_classes=n_classes,
        image_size=image_size,
    )


def materialize_batches(loader: DataLoader, max_horizon: int, device, dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for xb, yb in loader:
        out.append((xb.to(device=device, dtype=dtype), yb.to(device=device)))
        if len(out) >= max_horizon:
            break
    if out and len(out) < max_horizon:
        base = list(out)
        i = 0
        while len(out) < max_horizon:
            xb, yb = base[i % len(base)]
            out.append((xb.clone(), yb.clone()))
            i += 1
    if len(out) < max_horizon:
        raise RuntimeError(f"needed {max_horizon} continuation batches, got {len(out)}")
    return out
