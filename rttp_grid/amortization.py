from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import GridConfig
from .data import make_task_data, materialize_batches
from .models import FunctionalModel, build_model, load_buffers
from .runner import (
    BoundaryState,
    configure_functional_model,
    make_context,
    sgd_step,
    sgd_tangent_step,
    train_or_load_boundary,
)
from .utils import atomic_json, atomic_write_csv, cosine, duration, rel_error, seed_everything


@dataclass(frozen=True)
class AmortizationConfig:
    run_name: str = "rttp_amortization_cifar"
    root_name: str = "rttp_amortization_cifar"
    dataset: str = "split_cifar10"
    architecture: str = "smallcnn"
    seed: int = 0
    horizon: int = 40
    learning_rate: float = 0.03
    momentum: float = 0.9
    radius_scale: float = 1e-3
    response_rank: int = 8
    candidate_counts: tuple[int, ...] = (5, 10, 20, 50)
    repeats: int = 3
    warmup: int = 1
    cifar_train_per_class: int = 1000
    cifar_test_per_class: int = 250
    batch_size: int = 128
    boundary_epochs_smallcnn_cifar: int = 12
    boundary_epochs_resnet18_cifar: int = 20
    num_workers: int = 2
    force_boundary: bool = False
    drive_base: str = "/content/drive/MyDrive"
    use_google_drive: bool = False
    dtype: str = "float32"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _allocated_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.memory_allocated(device) / (1024**2))


def _time_block(device: torch.device, fn):
    _sync(device)
    _reset_peak_memory(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    elapsed = time.perf_counter() - start
    return value, elapsed, _peak_memory_mb(device), _allocated_memory_mb(device)


def _make_grid_config(cfg: AmortizationConfig) -> GridConfig:
    return GridConfig(
        run_name=cfg.run_name,
        root_name=cfg.root_name,
        drive_base=cfg.drive_base,
        use_google_drive=cfg.use_google_drive,
        datasets=(cfg.dataset,),
        architectures=(cfg.architecture,),
        optimizers=("heavy_ball",),
        seeds=(cfg.seed,),
        cifar_train_per_class=cfg.cifar_train_per_class,
        cifar_test_per_class=cfg.cifar_test_per_class,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        boundary_epochs_smallcnn_cifar=cfg.boundary_epochs_smallcnn_cifar,
        boundary_epochs_resnet18_cifar=cfg.boundary_epochs_resnet18_cifar,
        horizons_sgd=(cfg.horizon,),
        sgd_response_lrs_cifar=(cfg.learning_rate,),
        sgd_momenta=(cfg.momentum,),
        run_preflight_audit=False,
        run_dataset_probe=False,
        run_boundary=True,
        run_response=False,
        run_frozen_baseline=False,
        run_summary=False,
        run_figures=False,
        dtype=cfg.dtype,
        force=cfg.force_boundary,
    )


def _basis(theta: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 314159)
    q = torch.randn((int(rank), theta.numel()), generator=gen, dtype=theta.dtype)
    q = q.to(device=theta.device)
    q = torch.linalg.qr(q.T, mode="reduced").Q.T.contiguous()
    radius_base = theta.norm().clamp_min(1e-12)
    return radius_base * q


def _candidate_coefficients(rank: int, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 271828 + int(count))
    alpha = torch.randn((int(count), int(rank)), generator=gen, dtype=torch.float32)
    alpha = alpha / alpha.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return alpha


def _continue_sgd(
    fmodel: FunctionalModel,
    theta0: torch.Tensor,
    m0: torch.Tensor,
    batches,
    lr: float,
    mu: float,
    wd: float,
) -> torch.Tensor:
    theta, m = theta0.clone(), m0.clone()
    for xb, yb in batches:
        theta, m = sgd_step(fmodel, theta, m, xb, yb, lr, mu, wd)
    return theta.detach()


def _response_columns(
    fmodel: FunctionalModel,
    theta0: torch.Tensor,
    basis: torch.Tensor,
    batches,
    lr: float,
    mu: float,
    wd: float,
) -> torch.Tensor:
    cols = []
    for b in basis:
        theta, m = theta0.clone(), torch.zeros_like(theta0)
        dtheta, dm = torch.zeros_like(theta0), b.clone()
        for xb, yb in batches:
            theta, m, dtheta, dm = sgd_tangent_step(fmodel, theta, m, dtheta, dm, xb, yb, lr, mu, wd)
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _evaluate_response_candidates(response: torch.Tensor, alpha: torch.Tensor, scale: float) -> torch.Tensor:
    return float(scale) * alpha.to(device=response.device, dtype=response.dtype).matmul(response)


def _replay_candidates(
    fmodel: FunctionalModel,
    theta0: torch.Tensor,
    basis: torch.Tensor,
    alpha: torch.Tensor,
    batches,
    lr: float,
    mu: float,
    wd: float,
    scale: float,
) -> torch.Tensor:
    endpoints = []
    alpha = alpha.to(device=theta0.device, dtype=theta0.dtype)
    for row in alpha:
        m0 = float(scale) * row.matmul(basis)
        endpoints.append(_continue_sgd(fmodel, theta0, m0, batches, lr, mu, wd))
    return torch.stack(endpoints, dim=0)


def _summarize_timing(raw: pd.DataFrame, cfg: AmortizationConfig) -> pd.DataFrame:
    rows = []
    for k in cfg.candidate_counts:
        sub = raw[raw["candidate_count"].eq(k)]
        replay = sub[sub["phase"].eq("replay_candidates")]["seconds"]
        response_build = sub[sub["phase"].eq("response_build")]["seconds"]
        response_eval = sub[sub["phase"].eq("response_candidate_eval")]["seconds"]
        replay_mean = float(replay.mean())
        response_build_mean = float(response_build.mean())
        response_eval_mean = float(response_eval.mean())
        response_total = response_build_mean + response_eval_mean
        per_candidate_replay = replay_mean / k
        per_candidate_eval = response_eval_mean / k
        denom = per_candidate_replay - per_candidate_eval
        break_even = response_build_mean / denom if denom > 0 else math.inf
        rows.append(
            {
                "candidate_count": int(k),
                "horizon": int(cfg.horizon),
                "response_rank": int(cfg.response_rank),
                "repeats": int(cfg.repeats),
                "replay_seconds_mean": replay_mean,
                "response_build_seconds_mean": response_build_mean,
                "response_eval_seconds_mean": response_eval_mean,
                "response_total_seconds_mean": response_total,
                "replay_seconds_per_candidate": per_candidate_replay,
                "response_eval_seconds_per_candidate": per_candidate_eval,
                "wall_clock_savings_seconds": replay_mean - response_total,
                "measured_speedup": replay_mean / response_total if response_total > 0 else math.inf,
                "break_even_candidate_count": break_even,
                "replay_peak_memory_mb_mean": float(sub[sub["phase"].eq("replay_candidates")]["peak_memory_mb"].mean()),
                "response_peak_memory_mb_mean": float(sub[sub["phase"].isin(["response_build", "response_candidate_eval"])]["peak_memory_mb"].max()),
            }
        )
    return pd.DataFrame(rows)


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{rccccc}",
        "\\toprule",
        "Candidates & Replay & Response & Speedup & Measured replay & Measured response \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{int(row.candidate_count)} & "
            f"${int(row.candidate_count)}H$ & "
            f"$R_H B + {int(row.candidate_count)}$ & "
            f"${row.measured_speedup:.2f}\\times$ & "
            f"{row.replay_seconds_mean:.2f}s & "
            f"{row.response_total_seconds_mean:.2f}s \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write_report(root: Path, cfg: AmortizationConfig, summary: pd.DataFrame) -> None:
    best = summary.loc[summary["measured_speedup"].idxmax()]
    first = summary.iloc[0]
    report = f"""# Amortized Cost Timing Report

This CIFAR-only timing run measures the application scenario described in the
paper subsection: many nearby optimizer-state interventions evaluated at one
training boundary.

## Configuration

- dataset: `{cfg.dataset}`
- architecture: `{cfg.architecture}`
- optimizer: heavy-ball SGD momentum
- seed: `{cfg.seed}`
- horizon: `{cfg.horizon}`
- response rank: `{cfg.response_rank}`
- perturbation radius scale: `{cfg.radius_scale}`
- candidate counts: `{cfg.candidate_counts}`
- repeats: `{cfg.repeats}` after `{cfg.warmup}` warmup repeat(s)

## Interpretation

Replay times actually run one nonlinear continuation per candidate. Response
times build the shared finite-horizon response for the reduced basis once, then
evaluate candidate displacements by matrix multiplication in the reduced space.

At K={int(best.candidate_count)}, measured speedup is {best.measured_speedup:.2f}x.
The estimated break-even point from the measured per-candidate replay and
response-evaluation costs is {first.break_even_candidate_count:.2f} candidates
for the first table row.

Use `tables/amortization_summary.csv` for the paper numbers and
`tables/amortization_table.tex` as a drop-in LaTeX starting point.
"""
    (root / "README.md").write_text(report, encoding="utf-8")


def run_amortization_experiment(cfg: AmortizationConfig) -> Path:
    grid_cfg = _make_grid_config(cfg)
    ctx = make_context(grid_cfg)
    seed_everything(cfg.seed)

    timing_root = ctx.root
    tab = timing_root / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    atomic_json(asdict(cfg), timing_root / "amortization_config.json")

    data = make_task_data(grid_cfg, cfg.dataset, cfg.seed)
    boundary = train_or_load_boundary(ctx, cfg.dataset, cfg.architecture, "heavy_ball", cfg.seed, data)
    _, fmodel = configure_functional_model(ctx, cfg.architecture, data.n_classes, boundary.buffers)
    batches = materialize_batches(data.B_train, cfg.horizon, ctx.device, ctx.dtype)
    theta0 = boundary.theta.detach().clone()
    basis = _basis(theta0, cfg.response_rank, cfg.seed)
    alpha_by_k = {k: _candidate_coefficients(cfg.response_rank, k, cfg.seed) for k in cfg.candidate_counts}

    def response_build():
        return _response_columns(
            fmodel,
            theta0,
            basis,
            batches,
            cfg.learning_rate,
            cfg.momentum,
            grid_cfg.boundary_weight_decay,
        )

    
    for _ in range(cfg.warmup):
        response = response_build()
        _evaluate_response_candidates(response, alpha_by_k[min(cfg.candidate_counts)], cfg.radius_scale)
        _replay_candidates(
            fmodel,
            theta0,
            basis,
            alpha_by_k[min(cfg.candidate_counts)],
            batches,
            cfg.learning_rate,
            cfg.momentum,
            grid_cfg.boundary_weight_decay,
            cfg.radius_scale,
        )

    rows = []
    quality_rows = []
    for repeat in range(cfg.repeats):
        response, seconds, peak_mb, allocated_mb = _time_block(ctx.device, response_build)
        for k in cfg.candidate_counts:
            rows.append(
                {
                    "repeat": repeat,
                    "phase": "response_build",
                    "candidate_count": int(k),
                    "seconds": seconds,
                    "peak_memory_mb": peak_mb,
                    "allocated_memory_mb_after": allocated_mb,
                    "horizon": int(cfg.horizon),
                    "response_rank": int(cfg.response_rank),
                }
            )
            alpha = alpha_by_k[k]
            pred, eval_seconds, eval_peak_mb, eval_allocated_mb = _time_block(
                ctx.device,
                lambda alpha=alpha: _evaluate_response_candidates(response, alpha, cfg.radius_scale),
            )
            rows.append(
                {
                    "repeat": repeat,
                    "phase": "response_candidate_eval",
                    "candidate_count": int(k),
                    "seconds": eval_seconds,
                    "peak_memory_mb": eval_peak_mb,
                    "allocated_memory_mb_after": eval_allocated_mb,
                    "horizon": int(cfg.horizon),
                    "response_rank": int(cfg.response_rank),
                }
            )
            replay, replay_seconds, replay_peak_mb, replay_allocated_mb = _time_block(
                ctx.device,
                lambda alpha=alpha: _replay_candidates(
                    fmodel,
                    theta0,
                    basis,
                    alpha,
                    batches,
                    cfg.learning_rate,
                    cfg.momentum,
                    grid_cfg.boundary_weight_decay,
                    cfg.radius_scale,
                ),
            )
            rows.append(
                {
                    "repeat": repeat,
                    "phase": "replay_candidates",
                    "candidate_count": int(k),
                    "seconds": replay_seconds,
                    "peak_memory_mb": replay_peak_mb,
                    "allocated_memory_mb_after": replay_allocated_mb,
                    "horizon": int(cfg.horizon),
                    "response_rank": int(cfg.response_rank),
                }
            )
            nominal = _continue_sgd(
                fmodel,
                theta0,
                torch.zeros_like(theta0),
                batches,
                cfg.learning_rate,
                cfg.momentum,
                grid_cfg.boundary_weight_decay,
            )
            true_delta = replay - nominal.unsqueeze(0)
            quality_rows.append(
                {
                    "repeat": repeat,
                    "candidate_count": int(k),
                    "mean_endpoint_error": float(torch.stack([torch.tensor(rel_error(p, t)) for p, t in zip(pred, true_delta)]).nanmean()),
                    "mean_endpoint_cosine": float(torch.stack([torch.tensor(cosine(p, t)) for p, t in zip(pred, true_delta)]).nanmean()),
                    "mean_predicted_norm": float(pred.norm(dim=1).mean()),
                    "mean_true_delta_norm": float(true_delta.norm(dim=1).mean()),
                }
            )
            atomic_write_csv(pd.DataFrame(rows), tab / "amortization_timing_raw.csv")
            atomic_write_csv(pd.DataFrame(quality_rows), tab / "amortization_prediction_quality.csv")
            print(
                f"[amortization] repeat={repeat+1}/{cfg.repeats} K={k}: "
                f"replay={duration(replay_seconds)}, response_build={duration(seconds)}, "
                f"eval={eval_seconds:.4f}s",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    quality = pd.DataFrame(quality_rows)
    summary = _summarize_timing(raw, cfg)
    atomic_write_csv(raw, tab / "amortization_timing_raw.csv")
    atomic_write_csv(quality, tab / "amortization_prediction_quality.csv")
    atomic_write_csv(summary, tab / "amortization_summary.csv")
    (tab / "amortization_table.tex").write_text(_latex_table(summary), encoding="utf-8")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(ctx.device),
        "cuda_device_name": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "",
    }
    atomic_json(env, timing_root / "environment.json")
    _write_report(timing_root, cfg, summary)
    print(f"amortization outputs: {timing_root}")
    return timing_root

