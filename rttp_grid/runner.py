from __future__ import annotations

import gc
import json
import math
import platform
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats

from .audit import run_preflight_audit
from .config import GridConfig
from .data import make_task_data, materialize_batches
from .models import FunctionalModel, build_model, flatten_params, load_buffers, named_buffers_cpu, parameter_group_slices
from .utils import Eta, atomic_json, atomic_torch_save, atomic_write_csv, cosine, get_root, rel_error, seed_everything


@dataclass
class Context:
    cfg: GridConfig
    root: Path
    tab: Path
    cache: Path
    chunks: Path
    fig: Path
    audits: Path
    diagnostics: Path
    archives: Path
    device: torch.device
    dtype: torch.dtype


@dataclass
class BoundaryState:
    theta: torch.Tensor
    memory: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    step: int
    buffers: dict[str, torch.Tensor]
    losses: pd.DataFrame


@dataclass
class AdamState:
    theta: torch.Tensor
    m: torch.Tensor
    v: torch.Tensor
    step: int


@dataclass
class AdamTangent:
    dtheta: torch.Tensor
    dm: torch.Tensor
    dv: torch.Tensor


def make_context(cfg: GridConfig) -> Context:
    root = get_root(cfg)
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    ctx = Context(
        cfg=cfg,
        root=root,
        tab=root / "tables",
        cache=root / "cache",
        chunks=root / "cache" / "chunks",
        fig=root / "figures",
        audits=root / "audits",
        diagnostics=root / "diagnostics",
        archives=root / "archives",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=dtype,
    )
    for p in (ctx.root, ctx.tab, ctx.cache, ctx.chunks, ctx.fig, ctx.audits, ctx.diagnostics, ctx.archives):
        p.mkdir(parents=True, exist_ok=True)
    config_path = ctx.root / "config.json"
    if config_path.exists() and not cfg.force:
        previous = json.loads(config_path.read_text())
        controls = {"run_name", "root_name", "drive_base", "use_google_drive", "paper_material_dir", "force", "extra"}
        controls.update(k for k in asdict(cfg) if k.startswith("run_"))
        current = json.loads(json.dumps(asdict(cfg)))
        changed = [k for k in current if k not in controls and previous.get(k) != current[k]]
        if changed:
            raise ValueError("Existing output uses different scientific settings: " + ", ".join(changed) + ". Choose a new root_name or set force=True.")
    cfg.to_json(config_path)
    atomic_json(
        {
            "run_name": cfg.run_name,
            "root": str(ctx.root),
            "device": str(ctx.device),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "scientific_object": "finite-horizon optimizer-state response over dataset x architecture x optimizer grid",
            "grid": {
                "datasets": cfg.datasets,
                "architectures": cfg.architectures,
                "optimizers": cfg.optimizers,
                "seeds": cfg.seeds,
            },
        },
        ctx.root / "manifest.json",
    )
    try:
        (ctx.root / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    except Exception as exc:
        (ctx.root / "pip_freeze.txt").write_text(str(exc))
    return ctx


def probe_datasets(ctx: Context) -> pd.DataFrame:
    rows = []
    for dataset in ctx.cfg.datasets:
        if dataset == "split_cifar10":
            dataset_id = ctx.cfg.cifar10_dataset_id
            task_a = ctx.cfg.cifar10_task_a
            task_b = ctx.cfg.cifar10_task_b
        elif dataset == "split_tinyimagenet":
            dataset_id = ctx.cfg.tinyimagenet_dataset_id
            task_a = ctx.cfg.tiny_task_a
            task_b = ctx.cfg.tiny_task_b
        else:
            raise ValueError(dataset)
        ds = load_dataset(dataset_id)
        split_names = list(ds.keys())
        first_split = "train" if "train" in ds else split_names[0]
        first = ds[first_split][0]
        image_key = "img" if "img" in first else "image" if "image" in first else ""
        label_key = "label" if "label" in first else "fine_label" if "fine_label" in first else "class" if "class" in first else ""
        rows.append(
            {
                "dataset": dataset,
                "dataset_id": dataset_id,
                "passed": bool(image_key and label_key and "train" in split_names),
                "blocking": True,
                "splits": ";".join(split_names),
                "train_rows": len(ds["train"]) if "train" in ds else np.nan,
                "image_key": image_key,
                "label_key": label_key,
                "task_a_classes": ";".join(map(str, task_a)),
                "task_b_classes": ";".join(map(str, task_b)),
                "n_task_a": len(task_a),
                "n_task_b": len(task_b),
                "detail": "dataset schema probe",
            }
        )
    probe = pd.DataFrame(rows)
    atomic_write_csv(probe, ctx.diagnostics / "dataset_probe.csv")
    failed = probe[~probe.passed.astype(bool)]
    if len(failed):
        raise AssertionError(f"dataset probe failed: {failed.to_dict('records')}")
    return probe


def boundary_epochs(cfg: GridConfig, dataset: str, arch: str) -> int:
    if dataset == "split_cifar10":
        return cfg.boundary_epochs_resnet18_cifar if arch == "resnet18" else cfg.boundary_epochs_smallcnn_cifar
    return cfg.boundary_epochs_resnet18_tiny if arch == "resnet18" else cfg.boundary_epochs_smallcnn_tiny


def boundary_lr(cfg: GridConfig, dataset: str, optimizer: str) -> float:
    if optimizer == "heavy_ball":
        return cfg.boundary_sgd_lr_tiny if dataset == "split_tinyimagenet" else cfg.boundary_sgd_lr_cifar
    return cfg.boundary_adam_lr_tiny if dataset == "split_tinyimagenet" else cfg.boundary_adam_lr_cifar


def response_lrs(cfg: GridConfig, dataset: str, optimizer: str) -> tuple[float, ...]:
    if optimizer == "heavy_ball":
        return cfg.sgd_response_lrs_tiny if dataset == "split_tinyimagenet" else cfg.sgd_response_lrs_cifar
    return cfg.adam_response_lrs_tiny if dataset == "split_tinyimagenet" else cfg.adam_response_lrs_cifar


def horizons(cfg: GridConfig, optimizer: str) -> tuple[int, ...]:
    return cfg.horizons_sgd if optimizer == "heavy_ball" else cfg.horizons_adam


def boundary_path(ctx: Context, dataset: str, arch: str, optimizer: str, seed: int) -> Path:
    return ctx.cache / f"boundary_dataset={dataset}_optimizer={optimizer}_arch={arch}_seed={seed}.pt"


def configure_functional_model(ctx: Context, arch: str, n_classes: int, buffers: dict[str, torch.Tensor] | None = None) -> tuple[torch.nn.Module, FunctionalModel]:
    model = build_model(arch, n_classes).to(device=ctx.device, dtype=ctx.dtype)
    if buffers:
        load_buffers(model, buffers)
    model.eval()
    return model, FunctionalModel(model)


def extract_sgd_momentum(model: torch.nn.Module, opt: torch.optim.Optimizer) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        state = opt.state.get(p, {})
        buf = state.get("momentum_buffer")
        parts.append(torch.zeros_like(p).reshape(-1) if buf is None else buf.detach().reshape(-1))
    return torch.cat(parts)


def extract_adam_memory(model: torch.nn.Module, opt: torch.optim.Optimizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    ms, vs, steps = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        state = opt.state.get(p, {})
        ms.append(state.get("exp_avg", torch.zeros_like(p)).detach().reshape(-1))
        vs.append(state.get("exp_avg_sq", torch.zeros_like(p)).detach().reshape(-1))
        step = state.get("step", 0)
        if torch.is_tensor(step):
            step = int(step.detach().cpu().item())
        steps.append(int(step))
    return torch.cat(ms), torch.cat(vs), max(steps) if steps else 0


def train_or_load_boundary(ctx: Context, dataset: str, arch: str, optimizer: str, seed: int, data) -> BoundaryState:
    path = boundary_path(ctx, dataset, arch, optimizer, seed)
    if path.exists() and not ctx.cfg.force:
        obj = torch.load(path, map_location=ctx.device)
        memory = obj["memory"]
        if isinstance(memory, tuple):
            memory = tuple(x.to(ctx.device, dtype=ctx.dtype) for x in memory)
        else:
            memory = memory.to(ctx.device, dtype=ctx.dtype)
        return BoundaryState(
            theta=obj["theta"].to(ctx.device, dtype=ctx.dtype),
            memory=memory,
            step=int(obj.get("step", 0)),
            buffers={k: v.to(ctx.device, dtype=ctx.dtype) if v.is_floating_point() else v.to(ctx.device) for k, v in obj.get("buffers", {}).items()},
            losses=pd.DataFrame(obj.get("losses", [])),
        )
    if not ctx.cfg.run_boundary:
        raise FileNotFoundError(f"boundary missing and run_boundary=False: {path}")

    seed_everything(seed)
    model = build_model(arch, data.n_classes).to(device=ctx.device, dtype=ctx.dtype)
    model.train()
    if optimizer == "heavy_ball":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=boundary_lr(ctx.cfg, dataset, optimizer),
            momentum=ctx.cfg.boundary_sgd_momentum,
            weight_decay=ctx.cfg.boundary_weight_decay,
            nesterov=False,
            dampening=0.0,
        )
    elif optimizer == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=boundary_lr(ctx.cfg, dataset, optimizer),
            betas=(ctx.cfg.adam_beta1, ctx.cfg.adam_beta2),
            eps=ctx.cfg.adam_eps,
            weight_decay=ctx.cfg.adam_weight_decay,
        )
    else:
        raise ValueError(optimizer)

    rows = []
    total = boundary_epochs(ctx.cfg, dataset, arch) * len(data.A_train)
    timer = Eta(total, f"boundary {dataset} {optimizer} {arch} seed={seed}")
    step = 0
    for epoch in range(boundary_epochs(ctx.cfg, dataset, arch)):
        for xb, yb in data.A_train:
            xb = xb.to(device=ctx.device, dtype=ctx.dtype)
            yb = yb.to(ctx.device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1
            rows.append({"dataset": dataset, "architecture": arch, "optimizer": optimizer, "seed": seed, "epoch": epoch, "step": step, "loss": float(loss.detach())})
            if step % 20 == 0 or step == total:
                timer.update(note=f"loss={float(loss.detach()):.4g}", n=20 if step % 20 == 0 else 1)
    theta = flatten_params(model).detach()
    buffers = named_buffers_cpu(model)
    if optimizer == "heavy_ball":
        memory = extract_sgd_momentum(model, opt).detach()
        opt_step = step
    else:
        m, v, opt_step = extract_adam_memory(model, opt)
        memory = (m.detach(), v.detach())
    losses = pd.DataFrame(rows)
    obj = {"theta": theta.cpu(), "memory": memory if isinstance(memory, tuple) else memory.cpu(), "step": int(opt_step), "buffers": buffers, "losses": rows}
    if isinstance(memory, tuple):
        obj["memory"] = (memory[0].cpu(), memory[1].cpu())
    atomic_torch_save(obj, path)
    atomic_write_csv(losses, ctx.tab / f"boundary_losses_dataset={dataset}_optimizer={optimizer}_arch={arch}_seed={seed}.csv")
    return BoundaryState(theta, memory, int(opt_step), {k: v.to(ctx.device) for k, v in buffers.items()}, losses)


def grad_only(fmodel: FunctionalModel, theta: torch.Tensor, xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    th = theta.detach().requires_grad_(True)
    loss = fmodel.loss(th, xb, yb)
    g = torch.autograd.grad(loss, th)[0]
    return loss.detach(), g.detach()


def grad_hvp(fmodel: FunctionalModel, theta: torch.Tensor, vec: torch.Tensor, xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    th = theta.detach().requires_grad_(True)
    loss = fmodel.loss(th, xb, yb)
    g = torch.autograd.grad(loss, th, create_graph=True)[0]
    hvp = torch.autograd.grad(torch.dot(g, vec), th)[0]
    return g.detach(), hvp.detach()


def sgd_step(fmodel: FunctionalModel, theta: torch.Tensor, m: torch.Tensor, xb, yb, lr: float, mu: float, wd: float) -> tuple[torch.Tensor, torch.Tensor]:
    _, g = grad_only(fmodel, theta, xb, yb)
    if wd:
        g = g + wd * theta
    m_next = mu * m + g
    return (theta - lr * m_next).detach(), m_next.detach()


def sgd_tangent_step(fmodel: FunctionalModel, theta, m, dtheta, dm, xb, yb, lr: float, mu: float, wd: float):
    _, hv = grad_hvp(fmodel, theta, dtheta, xb, yb)
    if wd:
        hv = hv + wd * dtheta
    dm_next = mu * dm + hv
    dtheta_next = dtheta - lr * dm_next
    theta_next, m_next = sgd_step(fmodel, theta, m, xb, yb, lr, mu, wd)
    return theta_next, m_next, dtheta_next.detach(), dm_next.detach()


def sgd_frozen_tangent_step(fmodel: FunctionalModel, theta_ref, dtheta, dm, xb_ref, yb_ref, lr: float, mu: float, wd: float):
    _, hv = grad_hvp(fmodel, theta_ref, dtheta, xb_ref, yb_ref)
    if wd:
        hv = hv + wd * dtheta
    dm_next = mu * dm + hv
    dtheta_next = dtheta - lr * dm_next
    return dtheta_next.detach(), dm_next.detach()


def adam_denom(ctx: Context, v_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    floor = ctx.cfg.adam_eps**2
    active = v_hat > floor
    sqrt_v = torch.sqrt(torch.clamp(v_hat, min=floor))
    dsqrt = torch.where(active, 0.5 / sqrt_v, torch.zeros_like(v_hat))
    return sqrt_v + ctx.cfg.adam_eps, dsqrt


def adam_step(ctx: Context, fmodel: FunctionalModel, state: AdamState, xb, yb, lr: float) -> AdamState:
    _, g = grad_only(fmodel, state.theta, xb, yb)
    if ctx.cfg.adam_weight_decay:
        g = g + ctx.cfg.adam_weight_decay * state.theta
    step = state.step + 1
    b1, b2 = ctx.cfg.adam_beta1, ctx.cfg.adam_beta2
    m_next = b1 * state.m + (1 - b1) * g
    v_next = b2 * state.v + (1 - b2) * g.square()
    m_hat = m_next / (1 - b1**step)
    v_hat = v_next / (1 - b2**step)
    denom, _ = adam_denom(ctx, v_hat)
    theta_next = state.theta - lr * m_hat / denom
    return AdamState(theta_next.detach(), m_next.detach(), v_next.detach(), step)


def adam_tangent_step(ctx: Context, fmodel: FunctionalModel, state: AdamState, tangent: AdamTangent, xb, yb, lr: float, restricted_no_dv: bool = False):
    g, hdt = grad_hvp(fmodel, state.theta, tangent.dtheta, xb, yb)
    if ctx.cfg.adam_weight_decay:
        g = g + ctx.cfg.adam_weight_decay * state.theta
        hdt = hdt + ctx.cfg.adam_weight_decay * tangent.dtheta
    step = state.step + 1
    b1, b2 = ctx.cfg.adam_beta1, ctx.cfg.adam_beta2
    m_next = b1 * state.m + (1 - b1) * g
    v_next = b2 * state.v + (1 - b2) * g.square()
    dm_next = b1 * tangent.dm + (1 - b1) * hdt
    dv_next = torch.zeros_like(tangent.dv) if restricted_no_dv else b2 * tangent.dv + 2 * (1 - b2) * g * hdt
    c1, c2 = 1 - b1**step, 1 - b2**step
    m_hat, v_hat = m_next / c1, v_next / c2
    dm_hat, dv_hat = dm_next / c1, dv_next / c2
    denom, dsqrt = adam_denom(ctx, v_hat)
    d_update = dm_hat / denom - m_hat * dsqrt * dv_hat / denom.square()
    dtheta_next = tangent.dtheta - lr * d_update
    theta_next = state.theta - lr * m_hat / denom
    return AdamState(theta_next.detach(), m_next.detach(), v_next.detach(), step), AdamTangent(dtheta_next.detach(), dm_next.detach(), dv_next.detach())


def adam_frozen_tangent_step(ctx: Context, fmodel: FunctionalModel, base: AdamState, tangent: AdamTangent, xb_ref, yb_ref, lr: float):
    g, hdt = grad_hvp(fmodel, base.theta, tangent.dtheta, xb_ref, yb_ref)
    if ctx.cfg.adam_weight_decay:
        g = g + ctx.cfg.adam_weight_decay * base.theta
        hdt = hdt + ctx.cfg.adam_weight_decay * tangent.dtheta
    step = base.step + 1
    b1, b2 = ctx.cfg.adam_beta1, ctx.cfg.adam_beta2
    m_next = b1 * base.m + (1 - b1) * g
    v_next = b2 * base.v + (1 - b2) * g.square()
    dm_next = b1 * tangent.dm + (1 - b1) * hdt
    dv_next = b2 * tangent.dv + 2 * (1 - b2) * g * hdt
    c1, c2 = 1 - b1**step, 1 - b2**step
    m_hat, v_hat = m_next / c1, v_next / c2
    dm_hat, dv_hat = dm_next / c1, dv_next / c2
    denom, dsqrt = adam_denom(ctx, v_hat)
    d_update = dm_hat / denom - m_hat * dsqrt * dv_hat / denom.square()
    dtheta_next = tangent.dtheta - lr * d_update
    return AdamTangent(dtheta_next.detach(), dm_next.detach(), dv_next.detach())


def unit_direction(dim: int, seed: int, device, dtype) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn(dim, generator=gen, dtype=dtype).to(device)
    return z / z.norm().clamp_min(1e-30)


def sgd_direction(theta: torch.Tensor, idx: int, seed: int) -> tuple[torch.Tensor, str, float]:
    base = max(float(theta.norm()), 1e-12)
    return base * unit_direction(theta.numel(), seed + 1000 * idx + 101, theta.device, theta.dtype), f"random_momentum_{idx}", base


def adam_direction(ctx: Context, fmodel: FunctionalModel, arch: str, component: str, idx: int, seed: int, base: AdamState):
    dim = base.theta.numel()
    z = torch.zeros(dim, device=ctx.device, dtype=ctx.dtype)
    if component == "m":
        radius = max(float(base.m.norm()), 1e-12)
        q = radius * unit_direction(dim, seed + 1000 * idx + 11, ctx.device, ctx.dtype)
        return q, f"random_m_{idx}", radius
    slices = parameter_group_slices(arch, fmodel.param_names, fmodel.param_specs, ctx.cfg.resnet_rms_groups)
    labels = list(slices)
    if component == "rms_global":
        return torch.ones(dim, device=ctx.device, dtype=ctx.dtype), "all_parameters", 1.0
    if component == "rms_layerwise":
        q = z.clone()
        label = labels[int(idx)]
        q[slices[label]] = 1.0
        return q, label, 1.0
    if component == "rms_random_layerwise":
        gen = torch.Generator(device="cpu").manual_seed(seed + 1000 * idx + 29)
        coeffs = torch.randn(len(labels), generator=gen, dtype=ctx.dtype)
        coeffs = coeffs / coeffs.norm().clamp_min(1e-30)
        q = z.clone()
        for c, label in zip(coeffs, labels):
            q[slices[label]] = c.to(ctx.device)
        return q, f"random_layerwise_{idx}", 1.0
    raise ValueError(component)


def adam_direction_count(ctx: Context, fmodel: FunctionalModel, arch: str, component: str) -> int:
    if component == "rms_global":
        return 1
    if component == "rms_layerwise":
        return len(parameter_group_slices(arch, fmodel.param_names, fmodel.param_specs, ctx.cfg.resnet_rms_groups))
    return ctx.cfg.n_directions


def adam_tangent_initial(base: AdamState, component: str, q: torch.Tensor) -> AdamTangent:
    z = torch.zeros_like(base.theta)
    if component == "m":
        return AdamTangent(z.clone(), q.clone(), z.clone())
    return AdamTangent(z.clone(), z.clone(), 2.0 * base.v * q)


def adam_perturbed_state(base: AdamState, component: str, q: torch.Tensor, scale: float) -> AdamState:
    if component == "m":
        return AdamState(base.theta.clone(), base.m + float(scale) * q, base.v.clone(), base.step)
    exponent = torch.clamp(2.0 * float(scale) * q, min=-20.0, max=20.0)
    return AdamState(base.theta.clone(), base.m.clone(), base.v * torch.exp(exponent), base.step)


def response_chunk(ctx: Context, dataset: str, optimizer: str, arch: str, seed: int, lr: float, key: str) -> Path:
    schema = "schema=v2_frozen" if ctx.cfg.run_frozen_baseline else "schema=v2"
    return ctx.chunks / f"response_{schema}_dataset={dataset}_optimizer={optimizer}_arch={arch}_seed={seed}_lr={lr:g}_{key}.csv"


def run_sgd_setting(ctx: Context, dataset: str, arch: str, seed: int, boundary: BoundaryState, data, lr: float, mu: float) -> pd.DataFrame:
    key = f"mu={mu:g}"
    path = response_chunk(ctx, dataset, "heavy_ball", arch, seed, lr, key)
    if path.exists() and not ctx.cfg.force:
        return pd.read_csv(path)
    _, fmodel = configure_functional_model(ctx, arch, data.n_classes, boundary.buffers)
    batches = materialize_batches(data.B_train, max(ctx.cfg.horizons_sgd), ctx.device, ctx.dtype)
    theta0 = boundary.theta.detach().clone()
    m0 = torch.zeros_like(theta0)
    rows = []
    for direction in range(ctx.cfg.n_directions):
        coord, label, radius_base = sgd_direction(theta0, direction, seed)
        theta_nom, m_nom = theta0.clone(), m0.clone()
        tangent_by_h, frozen_tangent_by_h = {}, {}
        dtheta, dm = torch.zeros_like(theta0), coord.clone()
        fdtheta, fdm = torch.zeros_like(theta0), coord.clone()
        xb_ref, yb_ref = batches[0]
        for step, (xb, yb) in enumerate(batches, start=1):
            theta_nom, m_nom, dtheta, dm = sgd_tangent_step(fmodel, theta_nom, m_nom, dtheta, dm, xb, yb, lr, mu, ctx.cfg.boundary_weight_decay)
            if ctx.cfg.run_frozen_baseline:
                fdtheta, fdm = sgd_frozen_tangent_step(fmodel, theta0, fdtheta, fdm, xb_ref, yb_ref, lr, mu, ctx.cfg.boundary_weight_decay)
            if step in ctx.cfg.horizons_sgd:
                tangent_by_h[step] = dtheta.detach().clone()
                if ctx.cfg.run_frozen_baseline:
                    frozen_tangent_by_h[step] = fdtheta.detach().clone()
        
        nominal_snapshots = {}
        th, mm = theta0.clone(), m0.clone()
        for step, (xb, yb) in enumerate(batches, start=1):
            th, mm = sgd_step(fmodel, th, mm, xb, yb, lr, mu, ctx.cfg.boundary_weight_decay)
            if step in ctx.cfg.horizons_sgd:
                nominal_snapshots[step] = th.detach().clone()
        for scale in ctx.cfg.sgd_fd_scales:
            plus, pm = theta0.clone(), m0 + float(scale) * coord
            minus, nm = theta0.clone(), m0 - float(scale) * coord
            plus_snaps, minus_snaps = {}, {}
            for step, (xb, yb) in enumerate(batches, start=1):
                plus, pm = sgd_step(fmodel, plus, pm, xb, yb, lr, mu, ctx.cfg.boundary_weight_decay)
                minus, nm = sgd_step(fmodel, minus, nm, xb, yb, lr, mu, ctx.cfg.boundary_weight_decay)
                if step in ctx.cfg.horizons_sgd:
                    plus_snaps[step] = plus.detach().clone()
                    minus_snaps[step] = minus.detach().clone()
            for H in ctx.cfg.horizons_sgd:
                true_delta = plus_snaps[H] - nominal_snapshots[H]
                centered = (plus_snaps[H] - minus_snaps[H]) / (2 * float(scale))
                predictors = [("time_varying", tangent_by_h[H])]
                if ctx.cfg.run_frozen_baseline:
                    predictors.append(("frozen_boundary", frozen_tangent_by_h[H]))
                for response_model, tangent in predictors:
                    pred = float(scale) * tangent
                    rows.append(
                        {
                        "dataset": dataset,
                        "optimizer": "heavy_ball",
                        "architecture": arch,
                        "seed": seed,
                        "direction": direction,
                        "direction_label": label,
                        "learning_rate": lr,
                        "momentum": mu,
                        "memory_component": "momentum",
                        "response_model": response_model,
                        "state_parameterization": "additive_momentum",
                        "nominal_memory": "reset",
                        "step_offset": 0,
                        "horizon": H,
                        "radius_scale": scale,
                        "radius_base": radius_base,
                        "coordinate_norm": float(coord.norm()),
                        "tangent_norm": float(tangent.norm()),
                        "true_endpoint_norm": float(true_delta.norm()),
                        "centered_fd_norm": float(centered.norm()),
                        "endpoint_error": rel_error(pred, true_delta),
                        "endpoint_cosine": cosine(pred, true_delta),
                        "centered_error": rel_error(tangent, centered),
                        "centered_cosine": cosine(tangent, centered),
                        "batchnorm_buffers": "held_fixed" if arch == "resnet18" else "none",
                        }
                    )
    df = pd.DataFrame(rows)
    atomic_write_csv(df, path)
    return df


def run_adam_setting(ctx: Context, dataset: str, arch: str, seed: int, boundary: BoundaryState, data, lr: float, component: str) -> pd.DataFrame:
    _, fmodel = configure_functional_model(ctx, arch, data.n_classes, boundary.buffers)
    count = adam_direction_count(ctx, fmodel, arch, component)
    all_rows = []
    for direction in range(count):
        key = f"mem={component}_dir={direction}"
        path = response_chunk(ctx, dataset, "adam", arch, seed, lr, key)
        if path.exists() and not ctx.cfg.force:
            all_rows.append(pd.read_csv(path))
            continue
        m, v = boundary.memory
        base = AdamState(boundary.theta.detach().clone(), m.detach().clone(), v.detach().clone(), int(boundary.step))
        batches = materialize_batches(data.B_train, max(ctx.cfg.horizons_adam), ctx.device, ctx.dtype)
        q, label, radius_base = adam_direction(ctx, fmodel, arch, component, direction, seed, base)
        full_tangent_by_h, frozen_tangent_by_h, nominal_by_h = {}, {}, {}
        nominal = AdamState(base.theta.clone(), base.m.clone(), base.v.clone(), base.step)
        tangent = adam_tangent_initial(base, component, q)
        frozen_tangent = adam_tangent_initial(base, component, q)
        xb_ref, yb_ref = batches[0]
        for step, (xb, yb) in enumerate(batches, start=1):
            nominal, tangent = adam_tangent_step(ctx, fmodel, nominal, tangent, xb, yb, lr)
            if ctx.cfg.run_frozen_baseline:
                frozen_tangent = adam_frozen_tangent_step(ctx, fmodel, base, frozen_tangent, xb_ref, yb_ref, lr)
            if step in ctx.cfg.horizons_adam:
                full_tangent_by_h[step] = tangent.dtheta.detach().clone()
                if ctx.cfg.run_frozen_baseline:
                    frozen_tangent_by_h[step] = frozen_tangent.dtheta.detach().clone()
                nominal_by_h[step] = nominal.theta.detach().clone()
        rows = []
        scales = ctx.cfg.adam_m_fd_scales if component == "m" else ctx.cfg.adam_rms_fd_scales
        for scale in scales:
            plus = adam_perturbed_state(base, component, q, scale)
            minus = adam_perturbed_state(base, component, q, -scale)
            plus_snaps, minus_snaps = {}, {}
            for step, (xb, yb) in enumerate(batches, start=1):
                plus = adam_step(ctx, fmodel, plus, xb, yb, lr)
                minus = adam_step(ctx, fmodel, minus, xb, yb, lr)
                if step in ctx.cfg.horizons_adam:
                    plus_snaps[step] = plus.theta.detach().clone()
                    minus_snaps[step] = minus.theta.detach().clone()
            for H in ctx.cfg.horizons_adam:
                true_delta = plus_snaps[H] - nominal_by_h[H]
                centered = (plus_snaps[H] - minus_snaps[H]) / (2 * float(scale))
                predictors = [("time_varying", full_tangent_by_h[H])]
                if ctx.cfg.run_frozen_baseline:
                    predictors.append(("frozen_boundary", frozen_tangent_by_h[H]))
                for response_model, tangent_theta in predictors:
                    pred = float(scale) * tangent_theta
                    rows.append(
                        {
                        "dataset": dataset,
                        "optimizer": "adam",
                        "architecture": arch,
                        "seed": seed,
                        "direction": direction,
                        "direction_label": label,
                        "learning_rate": lr,
                        "beta1": ctx.cfg.adam_beta1,
                        "beta2": ctx.cfg.adam_beta2,
                        "adam_eps": ctx.cfg.adam_eps,
                        "weight_decay": ctx.cfg.adam_weight_decay,
                        "memory_component": component,
                        "tangent_model": "full_coupled",
                        "response_model": response_model,
                        "state_parameterization": "additive_first_moment" if component == "m" else "multiplicative_rms_memory",
                        "nominal_memory": "inherited",
                        "step_offset": int(boundary.step),
                        "horizon": H,
                        "radius_scale": scale,
                        "radius_base": radius_base,
                        "coordinate_norm": float(q.norm()),
                        "alpha": scale if component.startswith("rms_") else np.nan,
                        "rms_change_fraction": scale if component in ("rms_global", "rms_layerwise") else np.nan,
                        "max_abs_log_rms_change": float(scale * q.abs().max()) if component.startswith("rms_") else np.nan,
                        "tangent_norm": float(tangent_theta.norm()),
                        "true_endpoint_norm": float(true_delta.norm()),
                        "centered_fd_norm": float(centered.norm()),
                        "endpoint_error": rel_error(pred, true_delta),
                        "endpoint_cosine": cosine(pred, true_delta),
                        "centered_error": rel_error(tangent_theta, centered),
                        "centered_cosine": cosine(tangent_theta, centered),
                        "v_plus_nonnegative": bool((plus.v >= 0).all()),
                        "v_minus_nonnegative": bool((minus.v >= 0).all()),
                        "batchnorm_buffers": "held_fixed" if arch == "resnet18" else "none",
                        }
                    )
        df = pd.DataFrame(rows)
        atomic_write_csv(df, path)
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def expected_setting_count(cfg: GridConfig) -> int:
    total = 0
    for dataset in cfg.datasets:
        for arch in cfg.architectures:
            for seed in cfg.seeds:
                if "heavy_ball" in cfg.optimizers:
                    total += len(response_lrs(cfg, dataset, "heavy_ball")) * len(cfg.sgd_momenta)
                if "adam" in cfg.optimizers:
                    total += len(response_lrs(cfg, dataset, "adam")) * len(cfg.adam_memory_components)
    return total


def run_grid(cfg: GridConfig) -> Path:
    ctx = make_context(cfg)
    if cfg.run_preflight_audit:
        run_preflight_audit(ctx.audits)
    if cfg.run_dataset_probe:
        probe_datasets(ctx)
    all_rows, boundary_rows = [], []
    timer = Eta(expected_setting_count(cfg), "full RTTP grid settings")
    for dataset in cfg.datasets:
        for arch in cfg.architectures:
            for seed in cfg.seeds:
                data = make_task_data(cfg, dataset, seed)
                for optimizer in cfg.optimizers:
                    boundary = train_or_load_boundary(ctx, dataset, arch, optimizer, seed, data)
                    if optimizer == "heavy_ball":
                        mem_norm = float(boundary.memory.norm())
                        v_norm = np.nan
                    else:
                        mem_norm = float(boundary.memory[0].norm())
                        v_norm = float(boundary.memory[1].norm())
                    boundary_rows.append(
                        {
                            "dataset": dataset,
                            "architecture": arch,
                            "optimizer": optimizer,
                            "seed": seed,
                            "step": int(boundary.step),
                            "theta_norm": float(boundary.theta.norm()),
                            "memory_norm": mem_norm,
                            "v_norm": v_norm,
                        }
                    )
                    if not cfg.run_response:
                        continue
                    if optimizer == "heavy_ball":
                        for lr in response_lrs(cfg, dataset, optimizer):
                            for mu in cfg.sgd_momenta:
                                all_rows.append(run_sgd_setting(ctx, dataset, arch, seed, boundary, data, lr, mu))
                                timer.update(note=f"{dataset} {arch} seed={seed} SGD lr={lr:g} mu={mu:g}")
                    elif optimizer == "adam":
                        for lr in response_lrs(cfg, dataset, optimizer):
                            for component in cfg.adam_memory_components:
                                all_rows.append(run_adam_setting(ctx, dataset, arch, seed, boundary, data, lr, component))
                                timer.update(note=f"{dataset} {arch} seed={seed} Adam lr={lr:g} mem={component}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    boundary_summary = pd.DataFrame(boundary_rows)
    atomic_write_csv(boundary_summary, ctx.tab / "boundary_summary.csv")
    response_raw = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    atomic_write_csv(response_raw, ctx.tab / "response_raw.csv")
    if cfg.run_summary:
        summarize_outputs(ctx)
    if cfg.paper_material_dir:
        export_for_paper_material(ctx, Path(cfg.paper_material_dir))
    archive_run(ctx)
    return ctx.root


def ci95(values) -> tuple[float, float, float]:
    x = pd.Series(values).dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, np.nan, np.nan
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    q = float(stats.t.ppf(0.975, len(x) - 1))
    return mean, mean - q * se, mean + q * se


def _mean_ci(g: pd.DataFrame, cols: Sequence[str]) -> dict[str, float]:
    out = {}
    for col in cols:
        m, lo, hi = ci95(g[col])
        out[col] = m
        out[col + "_lo"] = lo
        out[col + "_hi"] = hi
    return out


def summarize_outputs(ctx_or_root) -> None:
    if isinstance(ctx_or_root, Context):
        ctx = ctx_or_root
    else:
        existing_root = Path(ctx_or_root).resolve()
        cfg = GridConfig.from_json(existing_root / "config.json")
        cfg = GridConfig(**{**asdict(cfg), "root_name": str(existing_root), "use_google_drive": False})
        ctx = make_context(cfg)
    raw_path = ctx.tab / "response_raw.csv"
    if not raw_path.exists():
        print("no response_raw.csv to summarize")
        return
    raw = pd.read_csv(raw_path)
    if not len(raw):
        return
    cols = ["endpoint_error", "endpoint_cosine", "centered_error", "centered_cosine", "tangent_norm", "true_endpoint_norm", "centered_fd_norm"]
    group_cols = ["dataset", "optimizer", "architecture", "seed", "learning_rate", "memory_component", "horizon", "radius_scale"]
    optional = [c for c in ("momentum", "beta1", "beta2", "tangent_model", "response_model") if c in raw.columns]
    seed_rows = []
    for keys, g in raw.groupby(group_cols + optional, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + optional, keys))
        row["n_directions"] = len(g)
        for col in cols:
            row[col] = float(g[col].mean())
        seed_rows.append(row)
    seed_level = pd.DataFrame(seed_rows)
    atomic_write_csv(seed_level, ctx.tab / "seed_level_response.csv")

    summary_group_cols = [c for c in group_cols if c != "seed"] + optional
    summary_rows = []
    for keys, g in seed_level.groupby(summary_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(summary_group_cols, keys))
        row["n_seeds"] = g.seed.nunique()
        row.update(_mean_ci(g, cols))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    atomic_write_csv(summary, ctx.tab / "response_summary_by_scale.csv")

    best_rows = []
    best_group = [c for c in summary_group_cols if c != "radius_scale"]
    for _, g in summary.groupby(best_group, dropna=False):
        gg = g.sort_values(["centered_error", "radius_scale"], ascending=[True, False])
        best_rows.append(gg.iloc[0].to_dict())
    best = pd.DataFrame(best_rows)
    atomic_write_csv(best, ctx.tab / "best_scale_summary.csv")
    paper = best.copy()
    atomic_write_csv(paper, ctx.tab / "paper_table_full_grid.csv")
    write_frozen_comparison_tables(ctx, summary, best)

    quality_rows = []
    def check(name, passed, blocking, detail):
        quality_rows.append({"check": name, "passed": bool(passed), "blocking": bool(blocking), "detail": str(detail)})
    check("response_raw_present", len(raw) > 0, True, f"rows={len(raw)}")
    if "response_model" in raw.columns:
        tv_raw = raw[raw.response_model.eq("time_varying")]
        frozen_raw = raw[raw.response_model.eq("frozen_boundary")]
    else:
        tv_raw = raw
        frozen_raw = raw.iloc[0:0]
    tv_finite = bool(np.isfinite(tv_raw[["endpoint_error", "centered_error"]].to_numpy(float)).all()) if len(tv_raw) else False
    check("finite_time_varying_metrics", tv_finite, True, "time-varying endpoint_error and centered_error finite")
    if len(frozen_raw):
        frozen_finite = bool(np.isfinite(frozen_raw[["endpoint_error", "centered_error"]].to_numpy(float)).all())
        bad_frozen = int((~np.isfinite(frozen_raw[["endpoint_error", "centered_error"]].to_numpy(float)).all(axis=1)).sum())
        check("finite_frozen_baseline_metrics", frozen_finite, False, f"nonfinite frozen rows={bad_frozen}; divergence is diagnostic, not blocking for RTTP")
    else:
        check("finite_key_metrics", tv_finite, True, "endpoint_error and centered_error finite")
    if "v_plus_nonnegative" in raw.columns:
        adam = raw[raw.optimizer.eq("adam")]
        if len(adam):
            check("adam_v_nonnegative", bool(adam.v_plus_nonnegative.all() and adam.v_minus_nonnegative.all()), True, "all Adam plus/minus v states nonnegative")
    for dataset in ctx.cfg.datasets:
        for opt in ctx.cfg.optimizers:
            subset = raw[(raw.dataset == dataset) & (raw.optimizer == opt)]
            check(f"block_present_{dataset}_{opt}", len(subset) > 0, True, f"rows={len(subset)}")
            for arch in ctx.cfg.architectures:
                arch_subset = subset[subset.architecture.eq(arch)]
                check(f"arch_present_{dataset}_{opt}_{arch}", len(arch_subset) > 0, True, f"rows={len(arch_subset)}")
                expected_horizons = set(horizons(ctx.cfg, opt))
                observed_horizons = set(int(x) for x in arch_subset.horizon.dropna().unique()) if len(arch_subset) else set()
                check(
                    f"horizons_complete_{dataset}_{opt}_{arch}",
                    expected_horizons.issubset(observed_horizons),
                    True,
                    f"expected={sorted(expected_horizons)}, observed={sorted(observed_horizons)}",
                )
                observed_seeds = set(int(x) for x in arch_subset.seed.dropna().unique()) if len(arch_subset) else set()
                expected_seeds = set(int(x) for x in ctx.cfg.seeds)
                check(
                    f"seeds_complete_{dataset}_{opt}_{arch}",
                    expected_seeds.issubset(observed_seeds),
                    True,
                    f"expected={sorted(expected_seeds)}, observed={sorted(observed_seeds)}",
                )
                if opt == "heavy_ball":
                    expected_components = {"momentum"}
                else:
                    expected_components = set(ctx.cfg.adam_memory_components)
                observed_components = set(str(x) for x in arch_subset.memory_component.dropna().unique()) if len(arch_subset) else set()
                check(
                    f"memory_components_complete_{dataset}_{opt}_{arch}",
                    expected_components.issubset(observed_components),
                    True,
                    f"expected={sorted(expected_components)}, observed={sorted(observed_components)}",
                )
                if ctx.cfg.run_frozen_baseline:
                    observed_models = set(str(x) for x in arch_subset.response_model.dropna().unique()) if "response_model" in arch_subset.columns and len(arch_subset) else set()
                    check(
                        f"response_models_complete_{dataset}_{opt}_{arch}",
                        {"time_varying", "frozen_boundary"}.issubset(observed_models),
                        True,
                        f"expected=['frozen_boundary', 'time_varying'], observed={sorted(observed_models)}",
                    )
    if ctx.cfg.run_preflight_audit:
        audit_path = ctx.audits / "preflight_audit_summary.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path)
            ok = bool(audit[audit.blocking.astype(bool)].passed.astype(bool).all())
            check("preflight_audit_passed", ok, True, audit.to_dict("records"))
        else:
            check("preflight_audit_passed", False, True, "preflight_audit_summary.csv missing")
    if ctx.cfg.run_dataset_probe:
        probe_path = ctx.diagnostics / "dataset_probe.csv"
        if probe_path.exists():
            probe = pd.read_csv(probe_path)
            ok = bool(probe[probe.blocking.astype(bool)].passed.astype(bool).all())
            check("dataset_probe_passed", ok, True, probe.to_dict("records"))
        else:
            check("dataset_probe_passed", False, True, "dataset_probe.csv missing")
    atomic_write_csv(pd.DataFrame(quality_rows), ctx.tab / "quality_report.csv")
    if ctx.cfg.run_figures:
        make_figures(ctx, best)


def write_frozen_comparison_tables(ctx: Context, summary: pd.DataFrame, best: pd.DataFrame) -> None:
    if "response_model" not in summary.columns:
        return
    key_cols = [
        c
        for c in [
            "dataset",
            "optimizer",
            "architecture",
            "learning_rate",
            "memory_component",
            "horizon",
            "radius_scale",
            "momentum",
            "beta1",
            "beta2",
            "tangent_model",
        ]
        if c in summary.columns
    ]
    value_cols = ["endpoint_error", "endpoint_cosine", "centered_error", "centered_cosine"]
    tv = summary[summary.response_model.eq("time_varying")][key_cols + value_cols].copy()
    fr = summary[summary.response_model.eq("frozen_boundary")][key_cols + value_cols].copy()
    if not len(tv) or not len(fr):
        return
    merged = tv.merge(fr, on=key_cols, suffixes=("_time_varying", "_frozen_boundary"))
    if not len(merged):
        return
    merged["endpoint_error_gap"] = merged["endpoint_error_frozen_boundary"] - merged["endpoint_error_time_varying"]
    merged["endpoint_error_gain"] = merged["endpoint_error_frozen_boundary"] / merged["endpoint_error_time_varying"].replace(0, np.nan)
    merged["centered_error_gap"] = merged["centered_error_frozen_boundary"] - merged["centered_error_time_varying"]
    merged["centered_error_gain"] = merged["centered_error_frozen_boundary"] / merged["centered_error_time_varying"].replace(0, np.nan)
    atomic_write_csv(merged, ctx.tab / "frozen_comparison_by_scale.csv")

    best_keys = [c for c in key_cols if c != "radius_scale"]
    selected = []
    for _, g in merged.groupby(best_keys, dropna=False):
        selected.append(g.sort_values(["centered_error_time_varying", "radius_scale"], ascending=[True, False]).iloc[0].to_dict())
    atomic_write_csv(pd.DataFrame(selected), ctx.tab / "frozen_best_comparison.csv")


def make_figures(ctx: Context, best: pd.DataFrame) -> None:
    if not len(best):
        return
    import matplotlib.pyplot as plt

    plot = best.copy().replace([np.inf, -np.inf], np.nan)
    plot = plot[np.isfinite(plot["endpoint_error"].to_numpy(float))]
    collapse_keys = [c for c in ["dataset", "optimizer", "architecture", "memory_component", "response_model", "horizon", "momentum"] if c in plot.columns]
    collapsed = []
    for _, gg in plot.groupby(collapse_keys, dropna=False):
        collapsed.append(gg.sort_values(["endpoint_error", "radius_scale"], ascending=[True, False]).iloc[0])
    plot = pd.DataFrame(collapsed) if collapsed else plot.iloc[0:0]
    for (dataset, optimizer, arch), g in plot.groupby(["dataset", "optimizer", "architecture"], dropna=False):
        plt.figure(figsize=(7.5, 4.8))
        if optimizer == "heavy_ball" and "momentum" in g.columns:
            g = g.copy()
            g["plot_group"] = g["momentum"].astype(str)
        else:
            g = g.copy()
            g["plot_group"] = g["memory_component"].astype(str)
        if "response_model" in g.columns:
            g["plot_group"] = g["plot_group"] + " / " + g["response_model"].astype(str)
        for label, gg in g.groupby("plot_group", dropna=False):
            gg = gg.sort_values("horizon")
            plt.semilogy(gg["horizon"], gg["endpoint_error"], marker="o", label=str(label))
        plt.xlabel("horizon H")
        plt.ylabel("relative endpoint error")
        plt.title(f"{dataset} {optimizer} {arch}")
        plt.legend(frameon=False)
        plt.tight_layout()
        stem = f"endpoint_error_by_horizon_dataset={dataset}_optimizer={optimizer}_arch={arch}"
        plt.savefig(ctx.fig / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.savefig(ctx.fig / f"{stem}.pdf", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7.5, 4.8))
        for label, gg in g.groupby("plot_group", dropna=False):
            gg = gg.sort_values("horizon")
            plt.plot(gg["horizon"], gg["endpoint_cosine"], marker="o", label=str(label))
        plt.xlabel("horizon H")
        plt.ylabel("endpoint displacement cosine")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{dataset} {optimizer} {arch}")
        plt.legend(frameon=False)
        plt.tight_layout()
        stem = f"cosine_by_horizon_dataset={dataset}_optimizer={optimizer}_arch={arch}"
        plt.savefig(ctx.fig / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.savefig(ctx.fig / f"{stem}.pdf", bbox_inches="tight")
        plt.close()

    h20 = plot[plot.horizon.eq(20) & plot.get("response_model", pd.Series("time_varying", index=plot.index)).eq("time_varying")].copy()
    if len(h20):
        h20["label"] = h20["dataset"].astype(str) + "\n" + h20["optimizer"].astype(str) + "\n" + h20["architecture"].astype(str) + "\n" + h20["memory_component"].astype(str)
        if "response_model" in h20.columns:
            h20["label"] = h20["label"] + "\n" + h20["response_model"].astype(str)
        plt.figure(figsize=(max(9, 0.45 * len(h20)), 5.2))
        plt.bar(range(len(h20)), h20["endpoint_error"])
        plt.xticks(range(len(h20)), h20["label"], rotation=60, ha="right")
        plt.ylabel("H=20 relative endpoint error")
        plt.title("RTTP full-grid H=20 comparison")
        plt.tight_layout()
        plt.savefig(ctx.fig / "full_grid_h20_endpoint_error.png", dpi=220, bbox_inches="tight")
        plt.savefig(ctx.fig / "full_grid_h20_endpoint_error.pdf", bbox_inches="tight")
        plt.close()

    frozen_path = ctx.tab / "frozen_best_comparison.csv"
    if frozen_path.exists():
        frozen = pd.read_csv(frozen_path)
        if len(frozen):
            for (dataset, optimizer, arch), g in frozen.groupby(["dataset", "optimizer", "architecture"], dropna=False):
                g = g.copy()
                if optimizer == "heavy_ball" and "momentum" in g.columns:
                    g["plot_group"] = "mu=" + g["momentum"].astype(str)
                else:
                    g["plot_group"] = g["memory_component"].astype(str)
                plt.figure(figsize=(7.5, 4.8))
                for label, gg in g.groupby("plot_group", dropna=False):
                    gg = gg.sort_values("horizon")
                    plt.plot(gg["horizon"], gg["endpoint_error_time_varying"], marker="o", label=f"{label} / time-varying")
                    plt.plot(gg["horizon"], gg["endpoint_error_frozen_boundary"], marker="x", linestyle="--", label=f"{label} / frozen")
                plt.xlabel("horizon H")
                plt.ylabel("relative endpoint error")
                plt.title(f"Frozen versus time-varying: {dataset} {optimizer} {arch}")
                plt.legend(frameon=False)
                plt.tight_layout()
                stem = f"frozen_vs_timevarying_dataset={dataset}_optimizer={optimizer}_arch={arch}"
                plt.savefig(ctx.fig / f"{stem}.png", dpi=220, bbox_inches="tight")
                plt.savefig(ctx.fig / f"{stem}.pdf", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(7.5, 4.8))
                for label, gg in g.groupby("plot_group", dropna=False):
                    gg = gg.sort_values("horizon")
                    plt.plot(gg["horizon"], gg["endpoint_error_gap"], marker="o", label=str(label))
                plt.axhline(0.0, color="black", linewidth=0.8)
                plt.xlabel("horizon H")
                plt.ylabel("frozen - time-varying endpoint error")
                plt.title(f"Frozen modeling gap: {dataset} {optimizer} {arch}")
                plt.legend(frameon=False)
                plt.tight_layout()
                stem = f"frozen_gap_dataset={dataset}_optimizer={optimizer}_arch={arch}"
                plt.savefig(ctx.fig / f"{stem}.png", dpi=220, bbox_inches="tight")
                plt.savefig(ctx.fig / f"{stem}.pdf", bbox_inches="tight")
                plt.close()


def export_for_paper_material(ctx: Context, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "figures", "audits", "diagnostics"):
        (destination / sub).mkdir(parents=True, exist_ok=True)
    for path in ctx.tab.glob("*.csv"):
        shutil.copy2(path, destination / "tables" / path.name)
    for path in ctx.fig.glob("*"):
        if path.is_file():
            shutil.copy2(path, destination / "figures" / path.name)
    for path in ctx.audits.glob("*.csv"):
        shutil.copy2(path, destination / "audits" / path.name)
    for path in ctx.diagnostics.glob("*.csv"):
        shutil.copy2(path, destination / "diagnostics" / path.name)
    for name in ("config.json", "manifest.json", "pip_freeze.txt"):
        src = ctx.root / name
        if src.exists():
            shutil.copy2(src, destination / name)
    readme = destination / "README.md"
    readme.write_text(
        "# RTTP Grid Paper-Material Export\n\n"
        f"Source root: `{ctx.root}`\n\n"
        "This export is derived from the reusable RTTP grid runner. "
        "Use `tables/quality_report.csv` before treating rows as paper evidence.\n"
    )


def archive_run(ctx: Context) -> Path:
    zip_path = ctx.archives / f"{ctx.cfg.run_name}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ctx.root.rglob("*"):
            if p.is_file() and ctx.archives not in p.parents:
                zf.write(p, p.relative_to(ctx.root.parent))
    print("output root:", ctx.root)
    print("archive:", zip_path)
    return zip_path
