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
import torch.nn.functional as F
from torch.func import functional_call

from .config import GridConfig
from .data import make_task_data, materialize_batches
from .runner import configure_functional_model, make_context, sgd_frozen_tangent_step, sgd_step, sgd_tangent_step, train_or_load_boundary
from .utils import atomic_json, atomic_write_csv, cosine, duration, rel_error, seed_everything


@dataclass(frozen=True)
class CandidateSelectionConfig:
    run_name: str = "rttp_cifar_candidate_selection"
    root_name: str = "rttp_cifar_candidate_selection"
    dataset: str = "split_cifar10"
    architecture: str = "smallcnn"
    seeds: tuple[int, ...] = (0,)
    horizon: int = 40
    learning_rate: float = 0.03
    momentum: float = 0.9
    radius_scale: float = 1e-3
    response_rank: int = 16
    candidate_count: int = 100
    objective_tradeoffs: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    topk_values: tuple[int, ...] = (5, 10)
    cifar_train_per_class: int = 1000
    cifar_test_per_class: int = 250
    eval_batches_per_task: int = 0
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


def _time_block(device: torch.device, fn):
    _sync(device)
    _reset_peak_memory(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - start, _peak_memory_mb(device)


def _grid_config(cfg: CandidateSelectionConfig) -> GridConfig:
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
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 424242)
    raw = torch.randn((int(rank), theta.numel()), generator=gen, dtype=theta.dtype).to(theta.device)
    q = torch.linalg.qr(raw.T, mode="reduced").Q.T.contiguous()
    return theta.norm().clamp_min(1e-12) * q


def _candidate_coefficients(rank: int, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7777)
    alpha = torch.randn((int(count), int(rank)), generator=gen, dtype=torch.float32)
    alpha = alpha / alpha.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return alpha


def _continue_sgd(fmodel, theta0, m0, batches, lr, mu, wd):
    theta, m = theta0.clone(), m0.clone()
    for xb, yb in batches:
        theta, m = sgd_step(fmodel, theta, m, xb, yb, lr, mu, wd)
    return theta.detach()


def _replay_endpoints(fmodel, theta0, basis, alpha, batches, lr, mu, wd, scale):
    alpha = alpha.to(device=theta0.device, dtype=theta0.dtype)
    endpoints = []
    for coeff in alpha:
        m0 = float(scale) * coeff.matmul(basis)
        endpoints.append(_continue_sgd(fmodel, theta0, m0, batches, lr, mu, wd))
    return torch.stack(endpoints, dim=0)


def _time_varying_columns(fmodel, theta0, basis, batches, lr, mu, wd):
    cols = []
    for b in basis:
        theta, m = theta0.clone(), torch.zeros_like(theta0)
        dtheta, dm = torch.zeros_like(theta0), b.clone()
        for xb, yb in batches:
            theta, m, dtheta, dm = sgd_tangent_step(fmodel, theta, m, dtheta, dm, xb, yb, lr, mu, wd)
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _frozen_columns(fmodel, theta0, basis, batches, lr, mu, wd):
    xb_ref, yb_ref = batches[0]
    cols = []
    for b in basis:
        dtheta, dm = torch.zeros_like(theta0), b.clone()
        for _ in batches:
            dtheta, dm = sgd_frozen_tangent_step(fmodel, theta0, dtheta, dm, xb_ref, yb_ref, lr, mu, wd)
        cols.append(dtheta.detach())
    return torch.stack(cols, dim=0)


def _predict_endpoints(theta_h: torch.Tensor, columns: torch.Tensor, alpha: torch.Tensor, scale: float):
    delta = float(scale) * alpha.to(device=columns.device, dtype=columns.dtype).matmul(columns)
    return theta_h.unsqueeze(0) + delta


def _materialize_eval(loader, device, dtype, max_batches: int):
    out = []
    for i, (xb, yb) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out.append((xb.to(device=device, dtype=dtype), yb.to(device=device)))
    return out


@torch.no_grad()
def _eval_theta(fmodel, theta: torch.Tensor, batches) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total = 0
    params = fmodel.vector_to_param_dict(theta)
    buffers = fmodel.buffer_dict()
    for xb, yb in batches:
        logits = functional_call(fmodel.model, (params, buffers), (xb,))
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.detach())
        total_correct += int((logits.argmax(dim=1) == yb).sum().detach())
        total += int(yb.numel())
    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
        "n_examples": total,
    }


def _evaluate_endpoint_set(fmodel, endpoints: torch.Tensor, A_eval, B_eval, label: str, seed: int):
    rows = []
    for i, theta in enumerate(endpoints):
        a = _eval_theta(fmodel, theta, A_eval)
        b = _eval_theta(fmodel, theta, B_eval)
        rows.append(
            {
                "seed": int(seed),
                "candidate_id": int(i),
                "predictor": label,
                "A_loss": a["loss"],
                "A_acc": a["acc"],
                "B_loss": b["loss"],
                "B_acc": b["acc"],
                "A_examples": int(a["n_examples"]),
                "B_examples": int(b["n_examples"]),
            }
        )
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = np.sign(a[i] - a[j])
            db = np.sign(b[i] - b[j])
            if da == 0 or db == 0:
                continue
            if da == db:
                concordant += 1
            else:
                discordant += 1
    den = concordant + discordant
    return float((concordant - discordant) / den) if den else float("nan")


def _ranking_metrics(scores: pd.DataFrame, cfg: CandidateSelectionConfig) -> pd.DataFrame:
    rows = []
    true = scores[scores["predictor"].eq("true_replay")].copy()
    for predictor in sorted(set(scores["predictor"]) - {"true_replay"}):
        pred = scores[scores["predictor"].eq(predictor)].copy()
        merged = true.merge(pred, on=["seed", "candidate_id", "tradeoff"], suffixes=("_true", "_pred"))
        for (seed, tradeoff), group in merged.groupby(["seed", "tradeoff"], dropna=False):
            true_score = group["score_true"].to_numpy(float)
            pred_score = group["score_pred"].to_numpy(float)
            candidate_ids = group["candidate_id"].to_numpy(int)
            true_best_idx = int(np.argmin(true_score))
            pred_best_idx = int(np.argmin(pred_score))
            true_best = int(candidate_ids[true_best_idx])
            selected = int(candidate_ids[pred_best_idx])
            row = {
                "seed": int(seed),
                "predictor": predictor,
                "tradeoff": float(tradeoff),
                "candidate_count": int(len(group)),
                "spearman": _spearman(pred_score, true_score),
                "kendall": _kendall(pred_score, true_score),
                "top1_agreement": bool(selected == true_best),
                "selected_candidate": selected,
                "true_best_candidate": true_best,
                "selected_true_score": float(true_score[pred_best_idx]),
                "true_best_score": float(true_score[true_best_idx]),
                "regret": float(true_score[pred_best_idx] - true_score[true_best_idx]),
                "selected_true_B_loss": float(group["B_loss_true"].to_numpy(float)[pred_best_idx]),
                "true_best_B_loss": float(group["B_loss_true"].to_numpy(float)[true_best_idx]),
                "selected_true_A_loss": float(group["A_loss_true"].to_numpy(float)[pred_best_idx]),
                "true_best_A_loss": float(group["A_loss_true"].to_numpy(float)[true_best_idx]),
            }
            for k in cfg.topk_values:
                kk = min(int(k), len(group))
                true_top = set(candidate_ids[np.argsort(true_score)[:kk]])
                pred_top = set(candidate_ids[np.argsort(pred_score)[:kk]])
                row[f"top{kk}_overlap_fraction"] = len(true_top & pred_top) / kk
            rows.append(row)
    return pd.DataFrame(rows)


def _score_rows(eval_rows: pd.DataFrame, cfg: CandidateSelectionConfig) -> pd.DataFrame:
    rows = []
    for _, row in eval_rows.iterrows():
        for tradeoff in cfg.objective_tradeoffs:
            rows.append(
                {
                    **row.to_dict(),
                    "tradeoff": float(tradeoff),
                    "score": float(row["B_loss"] + float(tradeoff) * row["A_loss"]),
                    "objective": f"B_loss_plus_{float(tradeoff):g}_A_loss",
                }
            )
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "n_seeds": ("seed", "nunique"),
        "spearman_mean": ("spearman", "mean"),
        "kendall_mean": ("kendall", "mean"),
        "top1_agreement_rate": ("top1_agreement", "mean"),
        "regret_mean": ("regret", "mean"),
        "regret_median": ("regret", "median"),
        "selected_true_score_mean": ("selected_true_score", "mean"),
        "true_best_score_mean": ("true_best_score", "mean"),
    }
    for col in sorted(c for c in metrics.columns if c.startswith("top") and c.endswith("_overlap_fraction")):
        agg[f"{col}_mean"] = (col, "mean")
    return (
        metrics.groupby(["predictor", "tradeoff"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["tradeoff", "regret_mean", "spearman_mean"], ascending=[True, True, False])
    )


def run_candidate_selection_experiment(cfg: CandidateSelectionConfig) -> Path:
    grid_cfg = _grid_config(cfg)
    ctx = make_context(grid_cfg)
    root = ctx.root
    tab = root / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    atomic_json(asdict(cfg), root / "candidate_selection_config.json")

    endpoint_quality_rows = []
    eval_rows = []
    timing_rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        data = make_task_data(grid_cfg, cfg.dataset, seed)
        boundary = train_or_load_boundary(ctx, cfg.dataset, cfg.architecture, "heavy_ball", seed, data)
        _, fmodel = configure_functional_model(ctx, cfg.architecture, data.n_classes, boundary.buffers)
        batches = materialize_batches(data.B_train, cfg.horizon, ctx.device, ctx.dtype)
        A_eval = _materialize_eval(data.A_test, ctx.device, ctx.dtype, cfg.eval_batches_per_task)
        B_eval = _materialize_eval(data.B_test, ctx.device, ctx.dtype, cfg.eval_batches_per_task)

        theta0 = boundary.theta.detach().clone()
        m0 = torch.zeros_like(theta0)
        basis = _basis(theta0, cfg.response_rank, seed)
        alpha = _candidate_coefficients(cfg.response_rank, cfg.candidate_count, seed)

        theta_h, nominal_seconds, nominal_peak = _time_block(
            ctx.device,
            lambda: _continue_sgd(fmodel, theta0, m0, batches, cfg.learning_rate, cfg.momentum, grid_cfg.boundary_weight_decay),
        )
        true_endpoints, replay_seconds, replay_peak = _time_block(
            ctx.device,
            lambda: _replay_endpoints(
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
        tv_cols, tv_seconds, tv_peak = _time_block(
            ctx.device,
            lambda: _time_varying_columns(
                fmodel, theta0, basis, batches, cfg.learning_rate, cfg.momentum, grid_cfg.boundary_weight_decay
            ),
        )
        frozen_cols, frozen_seconds, frozen_peak = _time_block(
            ctx.device,
            lambda: _frozen_columns(
                fmodel, theta0, basis, batches, cfg.learning_rate, cfg.momentum, grid_cfg.boundary_weight_decay
            ),
        )
        tv_endpoints, tv_eval_seconds, tv_eval_peak = _time_block(
            ctx.device, lambda: _predict_endpoints(theta_h, tv_cols, alpha, cfg.radius_scale)
        )
        frozen_endpoints, frozen_eval_seconds, frozen_eval_peak = _time_block(
            ctx.device, lambda: _predict_endpoints(theta_h, frozen_cols, alpha, cfg.radius_scale)
        )

        true_delta = true_endpoints - theta_h.unsqueeze(0)
        for label, endpoints, build_seconds, eval_seconds, peak in [
            ("time_varying_rttp", tv_endpoints, tv_seconds, tv_eval_seconds, max(tv_peak, tv_eval_peak)),
            ("frozen_boundary", frozen_endpoints, frozen_seconds, frozen_eval_seconds, max(frozen_peak, frozen_eval_peak)),
        ]:
            pred_delta = endpoints - theta_h.unsqueeze(0)
            for i, (pred, true) in enumerate(zip(pred_delta, true_delta)):
                endpoint_quality_rows.append(
                    {
                        "seed": int(seed),
                        "candidate_id": int(i),
                        "predictor": label,
                        "endpoint_error": rel_error(pred, true),
                        "endpoint_cosine": cosine(pred, true),
                        "predicted_delta_norm": float(pred.norm()),
                        "true_delta_norm": float(true.norm()),
                    }
                )
            timing_rows.append(
                {
                    "seed": int(seed),
                    "predictor": label,
                    "candidate_count": int(cfg.candidate_count),
                    "nominal_seconds": nominal_seconds,
                    "build_seconds": build_seconds,
                    "candidate_endpoint_eval_seconds": eval_seconds,
                    "total_prediction_seconds": nominal_seconds + build_seconds + eval_seconds,
                    "exhaustive_replay_seconds": replay_seconds,
                    "speedup_vs_exhaustive_replay": replay_seconds / max(nominal_seconds + build_seconds + eval_seconds, 1e-12),
                    "wall_clock_savings_seconds": replay_seconds - (nominal_seconds + build_seconds + eval_seconds),
                    "nominal_peak_memory_mb": nominal_peak,
                    "prediction_peak_memory_mb": peak,
                    "replay_peak_memory_mb": replay_peak,
                    "nonlinear_continuations_exhaustive": int(cfg.candidate_count),
                    "nonlinear_continuations_response": 1,
                }
            )

        for label, endpoints in [
            ("true_replay", true_endpoints),
            ("time_varying_rttp", tv_endpoints),
            ("frozen_boundary", frozen_endpoints),
        ]:
            eval_rows.extend(_evaluate_endpoint_set(fmodel, endpoints, A_eval, B_eval, label, seed))

        atomic_write_csv(pd.DataFrame(endpoint_quality_rows), tab / "candidate_endpoint_prediction_quality.csv")
        atomic_write_csv(pd.DataFrame(eval_rows), tab / "candidate_endpoint_eval_raw.csv")
        atomic_write_csv(pd.DataFrame(timing_rows), tab / "candidate_selection_timing.csv")
        print(
            f"[candidate selection] seed={seed} done | replay={duration(replay_seconds)} | "
            f"RTTP response={duration(tv_seconds)}",
            flush=True,
        )

    eval_df = pd.DataFrame(eval_rows)
    score_df = _score_rows(eval_df, cfg)
    metrics = _ranking_metrics(score_df, cfg)
    summary = _summary(metrics)
    timing = pd.DataFrame(timing_rows)
    quality = pd.DataFrame(endpoint_quality_rows)

    atomic_write_csv(eval_df, tab / "candidate_endpoint_eval_raw.csv")
    atomic_write_csv(score_df, tab / "candidate_scores_by_objective.csv")
    atomic_write_csv(metrics, tab / "candidate_ranking_metrics.csv")
    atomic_write_csv(summary, tab / "candidate_ranking_summary.csv")
    atomic_write_csv(timing, tab / "candidate_selection_timing.csv")
    atomic_write_csv(quality, tab / "candidate_endpoint_prediction_quality.csv")

    latex = summary.copy()
    for col in latex.select_dtypes(include=["number", "bool"]).columns:
        if col not in ("n_seeds",):
            latex[col] = latex[col].map(lambda x: f"{x:.4g}")
    (tab / "candidate_ranking_summary.tex").write_text(latex.to_latex(index=False), encoding="utf-8")

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(ctx.device),
        "cuda_device_name": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "",
    }
    atomic_json(env, root / "environment.json")
    readme = f"""# CIFAR Optimizer-State Candidate Selection

This run tests whether RTTP can select optimizer-state interventions
prospectively, without replaying every candidate.

For each seed, the experiment samples `{cfg.candidate_count}` nearby momentum
interventions in a rank-`{cfg.response_rank}` reduced subspace at the Task-A to
Task-B boundary. It compares rankings induced by predicted endpoints against
rankings from exhaustive nonlinear replay.

Objectives are `B_loss + lambda * A_loss` for:

```text
{cfg.objective_tradeoffs}
```

Primary outputs:

- `tables/candidate_scores_by_objective.csv`
- `tables/candidate_ranking_metrics.csv`
- `tables/candidate_ranking_summary.csv`
- `tables/candidate_selection_timing.csv`
- `tables/candidate_endpoint_prediction_quality.csv`

The key paper-facing quantities are Spearman/Kendall rank correlation,
Top-1 agreement, Top-k overlap, regret, and wall-clock savings versus exhaustive
replay.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(f"candidate selection outputs: {root}")
    return root
