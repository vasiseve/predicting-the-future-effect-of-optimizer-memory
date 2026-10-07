from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class Eta:
    def __init__(self, total: int, label: str):
        self.total = max(int(total), 1)
        self.label = label
        self.done = 0
        self.start = time.perf_counter()

    def update(self, note: str = "", n: int = 1) -> None:
        self.done += int(n)
        elapsed = time.perf_counter() - self.start
        rate = self.done / elapsed if elapsed > 0 else math.nan
        remaining = (self.total - self.done) / rate if rate and np.isfinite(rate) else math.nan
        eta = duration(remaining) if np.isfinite(remaining) else "unknown"
        suffix = f" | {note}" if note else ""
        print(f"[{self.label}] {self.done}/{self.total} | elapsed {duration(elapsed)} | ETA {eta}{suffix}", flush=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def rel_error(pred: torch.Tensor, true: torch.Tensor, tol: float = 1e-12) -> float:
    den = float(true.norm())
    if den < tol:
        return float("nan")
    return float((pred - true).norm() / den)


def cosine(pred: torch.Tensor, true: torch.Tensor, tol: float = 1e-12) -> float:
    pn, tn = float(pred.norm()), float(true.norm())
    if min(pn, tn) < tol:
        return float("nan")
    return float(torch.dot(pred.flatten(), true.flatten()) / (pred.norm() * true.norm()))


def get_root(cfg) -> Path:
    in_colab = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in str(os.environ.get("PYTHONPATH", ""))
    if in_colab and cfg.use_google_drive:
        return Path(cfg.drive_base) / cfg.root_name
    return Path.cwd() / cfg.root_name

