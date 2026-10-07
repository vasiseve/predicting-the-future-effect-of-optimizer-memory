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
from .runner import (
    configure_functional_model,
    make_context,
    sgd_frozen_tangent_step,
    sgd_step,
    sgd_tangent_step,
    train_or_load_boundary,
)
from .utils import atomic_json, atomic_write_csv, cosine, duration, rel_error, seed_everything


@dataclass(frozen=True)
class SurrogateBaselineConfig:
    run_name: str = "rttp_cifar_surrogate_baselines"
    root_name: str = "rttp_cifar_surrogate_baselines"
    dataset: str = "split_cifar10"
    architecture: str = "smallcnn"
    seeds: tuple[int, ...] = (0,)
    horizon: int = 40
    learning_rate: float = 0.03
    momentum: float = 0.9
    radius_scale: float = 1e-3
    response_rank: int = 16
    candidate_count: int = 32
    periodic_refresh_periods: tuple[int, ...] = (2, 5, 10)
    low_rank_response_ranks: tuple[int, ...] = (2, 4, 8)
    truncated_tail_lengths: tuple[int, ...] = (5, 10, 20)
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
    return value, time.perf_counter() - start, _peak_memory_mb(device), _allocated_memory_mb(device)


def _grid_config(cfg: SurrogateBaselineConfig) -> GridConfig:
    return GridConfig(
        run_name=cfg.run_name,
        root_name=cfg.root_name,
        drive_base=cfg.drive_base,
        use_google_drive=cfg.use_google_drive,
        datasets=(cfg.dataset,),
        architectures=(cfg.architecture,),
        optimizers=("heavy_ball",),
        seeds=cfg.seeds,
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
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 12345)
    raw = torch.randn((int(rank), theta.numel()), generator=gen, dtype=theta.dtype).to(theta.device)
    q = torch.linalg.qr(raw.T, mode="reduced").Q.T.contiguous()
    return theta.norm().clamp_min(1e-12) * q


def _candidate_coefficients(rank: int, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 54321)
    alpha = torch.randn((int(count), int(rank)), generator=gen, dtype=torch.float32)
    return alpha / alpha.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _nominal_snapshots(fmodel, theta0, m0, batches, lr, mu, wd):
    theta, m = theta0.clone(), m0.clone()
    theta_before, m_before = [], []
    theta_after, m_after = [], []
    for xb, yb in batches:
        theta_before.append(theta.detach().clone())
        m_before.append(m.detach().clone())
        theta, m = sgd_step(fmodel, theta, m, xb, yb, lr, mu, wd)
        theta_after.append(theta.detach().clone())
        m_after.append(m.detach().clone())
    return theta_before, m_before, theta_after, m_after


def _true_replay_endpoints(fmodel, theta0, basis, alpha, batches, lr, mu, wd, scale):
    alpha = alpha.to(device=theta0.device, dtype=theta0.dtype)
    endpoints = []
    for coeff in alpha:
        theta, m = theta0.clone(), float(scale) * coeff.matmul(basis)
        for xb, yb in batches:
            theta, m = sgd_step(fmodel, theta, m, xb, yb, lr, mu, wd)
        endpoints.append(theta.detach())
    return torch.stack(endpoints, dim=0)


def _full_time_varying_columns(fmodel, theta0, basis, batches, lr, mu, wd):
    cols = []
    for b in basis:
        theta, m = theta0.clone(), torch.zeros_like(theta0)
        dtheta, dm = torch.zeros_like(theta0), b.clone()
        for xb, yb in batches:
            theta, m, dtheta, dm = sgd_tangent_step(fmodel, theta, m, dtheta, dm, xb, yb, lr, mu, wd)
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _frozen_boundary_columns(fmodel, theta0, basis, batches, lr, mu, wd):
    xb_ref, yb_ref = batches[0]
    cols = []
    for b in basis:
        dtheta, dm = torch.zeros_like(theta0), b.clone()
        for _ in batches:
            dtheta, dm = sgd_frozen_tangent_step(fmodel, theta0, dtheta, dm, xb_ref, yb_ref, lr, mu, wd)
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _periodic_refresh_columns(fmodel, theta_before, basis, batches, lr, mu, wd, period: int):
    cols = []
    for b in basis:
        dtheta, dm = torch.zeros_like(b), b.clone()
        theta_ref, xb_ref, yb_ref = theta_before[0], batches[0][0], batches[0][1]
        for step, (xb, yb) in enumerate(batches):
            if step % int(period) == 0:
                theta_ref, xb_ref, yb_ref = theta_before[step], xb, yb
            dtheta, dm = sgd_frozen_tangent_step(fmodel, theta_ref, dtheta, dm, xb_ref, yb_ref, lr, mu, wd)
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _zero_hessian_columns(theta0, basis, horizon: int, lr: float, mu: float):
    cols = []
    for b in basis:
        dtheta, dm = torch.zeros_like(theta0), b.clone()
        for _ in range(int(horizon)):
            dm = mu * dm
            dtheta = dtheta - lr * dm
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _low_rank_columns(response: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(response, full_matrices=False)
    r = min(int(rank), s.numel())
    return (u[:, :r] * s[:r]).matmul(vh[:r, :]).detach()


def _hybrid_tail_predictions(
    fmodel,
    theta0,
    basis,
    alpha,
    theta_before,
    m_before,
    theta_after,
    batches,
    lr,
    mu,
    wd,
    scale,
    tail_length: int,
):
    horizon = len(batches)
    prefix = max(0, horizon - int(tail_length))
    alpha = alpha.to(device=theta0.device, dtype=theta0.dtype)
    preds = []
    for coeff in alpha:
        theta = theta0.clone()
        m = float(scale) * coeff.matmul(basis)
        for xb, yb in batches[:prefix]:
            theta, m = sgd_step(fmodel, theta, m, xb, yb, lr, mu, wd)
        dtheta = theta - theta_before[prefix]
        dm = m - m_before[prefix]
        nominal_theta = theta_before[prefix].clone()
        nominal_m = m_before[prefix].clone()
        for xb, yb in batches[prefix:]:
            nominal_theta, nominal_m, dtheta, dm = sgd_tangent_step(
                fmodel, nominal_theta, nominal_m, dtheta, dm, xb, yb, lr, mu, wd
            )
        preds.append((theta_after[-1] + dtheta).detach())
    return torch.stack(preds, dim=0)


def _predict_from_columns(columns: torch.Tensor, alpha: torch.Tensor, scale: float) -> torch.Tensor:
    return float(scale) * alpha.to(device=columns.device, dtype=columns.dtype).matmul(columns)


def _metric_rows(seed, surrogate, pred_delta, true_delta, build_seconds, eval_seconds, peak_memory_mb, extra=None):
    rows = []
    extra = extra or {}
    for i, (pred, true) in enumerate(zip(pred_delta, true_delta)):
        rows.append(
            {
                "seed": int(seed),
                "candidate_id": int(i),
                "surrogate": surrogate,
                "endpoint_error": rel_error(pred, true),
                "endpoint_cosine": cosine(pred, true),
                "predicted_norm": float(pred.norm()),
                "true_delta_norm": float(true.norm()),
                "build_seconds": float(build_seconds),
                "eval_seconds": float(eval_seconds),
                "total_seconds": float(build_seconds + eval_seconds),
                "peak_memory_mb": peak_memory_mb,
                **extra,
            }
        )
    return rows


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["surrogate"]
    optional = [c for c in ("period", "low_rank", "tail_length") if c in raw.columns]
    group_cols += optional
    return (
        raw.groupby(group_cols, dropna=False)
        .agg(
            n_candidates=("candidate_id", "count"),
            n_seeds=("seed", "nunique"),
            endpoint_error_mean=("endpoint_error", "mean"),
            endpoint_error_median=("endpoint_error", "median"),
            endpoint_cosine_mean=("endpoint_cosine", "mean"),
            predicted_norm_mean=("predicted_norm", "mean"),
            true_delta_norm_mean=("true_delta_norm", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
            build_seconds_mean=("build_seconds", "mean"),
            eval_seconds_mean=("eval_seconds", "mean"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
        )
        .reset_index()
        .sort_values(["endpoint_error_mean", "total_seconds_mean"])
    )


def run_surrogate_baseline_experiment(cfg: SurrogateBaselineConfig) -> Path:
    grid_cfg = _grid_config(cfg)
    ctx = make_context(grid_cfg)
    root = ctx.root
    tab = root / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    atomic_json(asdict(cfg), root / "surrogate_baseline_config.json")

    all_rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        data = make_task_data(grid_cfg, cfg.dataset, seed)
        boundary = train_or_load_boundary(ctx, cfg.dataset, cfg.architecture, "heavy_ball", seed, data)
        _, fmodel = configure_functional_model(ctx, cfg.architecture, data.n_classes, boundary.buffers)
        batches = materialize_batches(data.B_train, cfg.horizon, ctx.device, ctx.dtype)
        theta0 = boundary.theta.detach().clone()
        m0 = torch.zeros_like(theta0)
        basis = _basis(theta0, cfg.response_rank, seed)
        alpha = _candidate_coefficients(cfg.response_rank, cfg.candidate_count, seed)

        (theta_before, m_before, theta_after, _), nominal_seconds, nominal_peak, _ = _time_block(
            ctx.device,
            lambda: _nominal_snapshots(
                fmodel, theta0, m0, batches, cfg.learning_rate, cfg.momentum, grid_cfg.boundary_weight_decay
            ),
        )
        theta_h = theta_after[-1]
        true_endpoints, replay_seconds, replay_peak, _ = _time_block(
            ctx.device,
            lambda: _true_replay_endpoints(
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
        true_delta = true_endpoints - theta_h.unsqueeze(0)
        all_rows.extend(
            _metric_rows(
                seed,
                "exact_replay",
                true_delta,
                true_delta,
                0.0,
                replay_seconds,
                replay_peak,
                {"nominal_seconds": nominal_seconds, "candidate_count": cfg.candidate_count},
            )
        )

        columns, build_seconds, build_peak, _ = _time_block(
            ctx.device,
            lambda: _full_time_varying_columns(
                fmodel, theta0, basis, batches, cfg.learning_rate, cfg.momentum, grid_cfg.boundary_weight_decay
            ),
        )
        pred, eval_seconds, eval_peak, _ = _time_block(ctx.device, lambda: _predict_from_columns(columns, alpha, cfg.radius_scale))
        all_rows.extend(
            _metric_rows(
                seed,
                "time_varying_rttp",
                pred,
                true_delta,
                build_seconds,
                eval_seconds,
                max(build_peak, eval_peak),
                {"candidate_count": cfg.candidate_count, "response_rank": cfg.response_rank},
            )
        )

        frozen, fbuild, fpeak, _ = _time_block(
            ctx.device,
            lambda: _frozen_boundary_columns(
                fmodel, theta0, basis, batches, cfg.learning_rate, cfg.momentum, grid_cfg.boundary_weight_decay
            ),
        )
        fpred, feval, feval_peak, _ = _time_block(ctx.device, lambda: _predict_from_columns(frozen, alpha, cfg.radius_scale))
        all_rows.extend(
            _metric_rows(
                seed,
                "frozen_boundary",
                fpred,
                true_delta,
                fbuild,
                feval,
                max(fpeak, feval_peak),
                {"candidate_count": cfg.candidate_count, "response_rank": cfg.response_rank},
            )
        )

        zero, zbuild, zpeak, _ = _time_block(
            ctx.device,
            lambda: _zero_hessian_columns(theta0, basis, cfg.horizon, cfg.learning_rate, cfg.momentum),
        )
        zpred, zeval, zeval_peak, _ = _time_block(ctx.device, lambda: _predict_from_columns(zero, alpha, cfg.radius_scale))
        all_rows.extend(
            _metric_rows(
                seed,
                "zero_hessian_memory_decay",
                zpred,
                true_delta,
                zbuild,
                zeval,
                max(zpeak, zeval_peak),
                {"candidate_count": cfg.candidate_count, "response_rank": cfg.response_rank},
            )
        )

        for period in cfg.periodic_refresh_periods:
            pc, pbuild, ppeak, _ = _time_block(
                ctx.device,
                lambda period=period: _periodic_refresh_columns(
                    fmodel,
                    theta_before,
                    basis,
                    batches,
                    cfg.learning_rate,
                    cfg.momentum,
                    grid_cfg.boundary_weight_decay,
                    period,
                ),
            )
            ppred, peval, peval_peak, _ = _time_block(ctx.device, lambda pc=pc: _predict_from_columns(pc, alpha, cfg.radius_scale))
            all_rows.extend(
                _metric_rows(
                    seed,
                    "periodic_refresh",
                    ppred,
                    true_delta,
                    pbuild,
                    peval,
                    max(ppeak, peval_peak),
                    {
                        "period": int(period),
                        "candidate_count": cfg.candidate_count,
                        "response_rank": cfg.response_rank,
                    },
                )
            )

        for rank in cfg.low_rank_response_ranks:
            lr_cols, lr_build, lr_peak, _ = _time_block(ctx.device, lambda rank=rank: _low_rank_columns(columns, rank))
            lr_pred, lr_eval, lr_eval_peak, _ = _time_block(
                ctx.device, lambda lr_cols=lr_cols: _predict_from_columns(lr_cols, alpha, cfg.radius_scale)
            )
            all_rows.extend(
                _metric_rows(
                    seed,
                    "low_rank_response",
                    lr_pred,
                    true_delta,
                    build_seconds + lr_build,
                    lr_eval,
                    max(build_peak, lr_peak, lr_eval_peak),
                    {
                        "low_rank": int(rank),
                        "candidate_count": cfg.candidate_count,
                        "response_rank": cfg.response_rank,
                    },
                )
            )

        for tail in cfg.truncated_tail_lengths:
            hp, hseconds, hpeak, _ = _time_block(
                ctx.device,
                lambda tail=tail: _hybrid_tail_predictions(
                    fmodel,
                    theta0,
                    basis,
                    alpha,
                    theta_before,
                    m_before,
                    theta_after,
                    batches,
                    cfg.learning_rate,
                    cfg.momentum,
                    grid_cfg.boundary_weight_decay,
                    cfg.radius_scale,
                    tail,
                ),
            )
            all_rows.extend(
                _metric_rows(
                    seed,
                    "truncated_replay_tail_tangent",
                    hp - theta_h.unsqueeze(0),
                    true_delta,
                    0.0,
                    hseconds,
                    hpeak,
                    {
                        "tail_length": int(tail),
                        "candidate_count": cfg.candidate_count,
                        "response_rank": cfg.response_rank,
                    },
                )
            )

        raw = pd.DataFrame(all_rows)
        atomic_write_csv(raw, tab / "surrogate_baseline_raw.csv")
        atomic_write_csv(_summarize(raw), tab / "surrogate_baseline_summary.csv")
        print(
            f"[surrogate baselines] seed={seed} done | exact replay {duration(replay_seconds)} | "
            f"RTTP build {duration(build_seconds)}",
            flush=True,
        )

    raw = pd.DataFrame(all_rows)
    summary = _summarize(raw)
    atomic_write_csv(raw, tab / "surrogate_baseline_raw.csv")
    atomic_write_csv(summary, tab / "surrogate_baseline_summary.csv")

    exact_time = summary[summary["surrogate"].eq("exact_replay")]["eval_seconds_mean"].mean()
    summary_for_cost = summary.copy()
    summary_for_cost["wall_clock_savings_vs_replay"] = exact_time - summary_for_cost["total_seconds_mean"]
    summary_for_cost["speedup_vs_replay"] = exact_time / summary_for_cost["total_seconds_mean"].replace(0, np.nan)
    atomic_write_csv(summary_for_cost, tab / "surrogate_cost_quality_tradeoff.csv")

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(ctx.device),
        "cuda_device_name": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "",
    }
    atomic_json(env, root / "environment.json")

    readme = f"""# CIFAR Surrogate-Baseline Experiment

This run compares finite-horizon RTTP against simple surrogate predictors at
the same split-CIFAR boundary.

The target intervention is a reduced-basis perturbation of heavy-ball momentum.
The nonlinear replay endpoint is used only for evaluation.

Surrogates:

- `time_varying_rttp`: chronological finite-horizon tangent product.
- `frozen_boundary`: repeats the boundary Jacobian/batch for all steps.
- `periodic_refresh`: freezes a local Jacobian for short blocks and refreshes every `p` steps.
- `zero_hessian_memory_decay`: ignores curvature and only propagates momentum decay.
- `low_rank_response`: truncates the computed response map by SVD.
- `truncated_replay_tail_tangent`: replays each candidate for a prefix and linearizes only the remaining tail.

Primary outputs:

- `tables/surrogate_baseline_raw.csv`
- `tables/surrogate_baseline_summary.csv`
- `tables/surrogate_cost_quality_tradeoff.csv`

Configuration:

```json
{json.dumps(asdict(cfg), indent=2)}
```
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(f"surrogate baseline outputs: {root}")
    return root

