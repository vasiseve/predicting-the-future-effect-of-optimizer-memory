from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import atomic_write_csv, cosine, rel_error, seed_everything


def exact_optimizer_equations() -> pd.DataFrame:
    rows = [
        {
            "optimizer": "heavy_ball",
            "equation": "g_eff = grad L(theta) + wd*theta if wd>0 else grad L(theta); m_next = mu*m + g_eff; theta_next = theta - lr*m_next",
            "state": "x=(theta,m)",
            "memory_component": "momentum",
            "state_parameterization": "additive_momentum",
            "dampening": "0",
            "nesterov": "False",
            "weight_decay": "coupled L2 if wd>0",
            "scheduler": "none",
            "parameter_groups": "single implicit group",
        },
        {
            "optimizer": "adam",
            "equation": "g_eff = grad L(theta) + wd*theta if wd>0 else grad L(theta); m_next=beta1*m+(1-beta1)*g_eff; v_next=beta2*v+(1-beta2)*g_eff^2; theta_next=theta-lr*(m_next/c1)/(sqrt(clamp(v_next/c2,eps^2))+eps)",
            "state": "x=(theta,m,v,step)",
            "memory_component": "m,rms_global,rms_layerwise,rms_random_layerwise",
            "state_parameterization": "additive_first_moment or multiplicative_rms_memory",
            "bias_correction": "c1=1-beta1^step, c2=1-beta2^step",
            "amsgrad": "False",
            "maximize": "False",
            "weight_decay": "coupled L2 if wd>0; default wd=0",
            "parameter_groups": "single implicit group",
        },
    ]
    return pd.DataFrame(rows)


def _tiny_loss(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    w1 = theta[:12].view(3, 4)
    b1 = theta[12:15]
    w2 = theta[15:21].view(2, 3)
    b2 = theta[21:23]
    h = torch.tanh(x @ w1.T + b1)
    logits = h @ w2.T + b2
    return torch.nn.functional.cross_entropy(logits, y)


def _grad(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    th = theta.detach().requires_grad_(True)
    loss = _tiny_loss(th, x, y)
    return torch.autograd.grad(loss, th)[0].detach()


def _hvp(theta: torch.Tensor, vec: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    th = theta.detach().requires_grad_(True)
    loss = _tiny_loss(th, x, y)
    g = torch.autograd.grad(loss, th, create_graph=True)[0]
    return torch.autograd.grad(torch.dot(g, vec), th)[0].detach()


def _sgd_step(theta, m, x, y, lr, mu):
    g = _grad(theta, x, y)
    m_next = mu * m + g
    return (theta - lr * m_next).detach(), m_next.detach()


def _sgd_tangent(theta, m, dtheta, dm, x, y, lr, mu):
    hv = _hvp(theta, dtheta, x, y)
    dm_next = mu * dm + hv
    dtheta_next = dtheta - lr * dm_next
    theta_next, m_next = _sgd_step(theta, m, x, y, lr, mu)
    return theta_next, m_next, dtheta_next.detach(), dm_next.detach()


def _adam_denom(v_hat: torch.Tensor, eps: float):
    floor = eps**2
    active = v_hat > floor
    sqrt_v = torch.sqrt(torch.clamp(v_hat, min=floor))
    dsqrt = torch.where(active, 0.5 / sqrt_v, torch.zeros_like(v_hat))
    return sqrt_v + eps, dsqrt


def _adam_step(theta, m, v, step, x, y, lr, b1, b2, eps):
    g = _grad(theta, x, y)
    next_step = step + 1
    m_next = b1 * m + (1 - b1) * g
    v_next = b2 * v + (1 - b2) * g.square()
    m_hat = m_next / (1 - b1**next_step)
    v_hat = v_next / (1 - b2**next_step)
    denom, _ = _adam_denom(v_hat, eps)
    theta_next = theta - lr * m_hat / denom
    return theta_next.detach(), m_next.detach(), v_next.detach(), next_step


def _adam_tangent(theta, m, v, step, dtheta, dm, dv, x, y, lr, b1, b2, eps):
    g = _grad(theta, x, y)
    hv = _hvp(theta, dtheta, x, y)
    next_step = step + 1
    m_next = b1 * m + (1 - b1) * g
    v_next = b2 * v + (1 - b2) * g.square()
    dm_next = b1 * dm + (1 - b1) * hv
    dv_next = b2 * dv + 2 * (1 - b2) * g * hv
    c1, c2 = 1 - b1**next_step, 1 - b2**next_step
    m_hat, v_hat = m_next / c1, v_next / c2
    dm_hat, dv_hat = dm_next / c1, dv_next / c2
    denom, dsqrt = _adam_denom(v_hat, eps)
    d_update = dm_hat / denom - m_hat * dsqrt * dv_hat / denom.square()
    dtheta_next = dtheta - lr * d_update
    theta_next = theta - lr * m_hat / denom
    return theta_next.detach(), m_next.detach(), v_next.detach(), next_step, dtheta_next.detach(), dm_next.detach(), dv_next.detach()


def run_preflight_audit(output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(123)
    dtype = torch.float64
    x = torch.randn(5, 4, dtype=dtype)
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    theta = torch.randn(23, dtype=dtype) * 0.2
    u = torch.randn_like(theta)
    rows = []

    
    dense_h = torch.autograd.functional.hessian(lambda th: _tiny_loss(th, x, y), theta)
    dense_hu = dense_h @ u
    direct_hu = _hvp(theta, u, x, y)
    rows.append(
        {
            "section": "hvp",
            "check": "dense_vs_direct_hvp",
            "passed": rel_error(direct_hu, dense_hu, 1e-30) < 1e-10,
            "blocking": True,
            "relative_error": rel_error(direct_hu, dense_hu, 1e-30),
            "cosine": cosine(direct_hu, dense_hu, 1e-30),
            "detail": "dense Hessian times vector equals autograd HVP",
        }
    )

    
    lr, mu = 0.07, 0.8
    m = torch.randn_like(theta) * 0.1
    dtheta = torch.randn_like(theta) * 0.01
    dm = torch.randn_like(theta) * 0.01
    _, _, manual_dt, manual_dm = _sgd_tangent(theta, m, dtheta, dm, x, y, lr, mu)
    best = math.inf
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        tp, mp = _sgd_step(theta + eps * dtheta, m + eps * dm, x, y, lr, mu)
        tm, mm = _sgd_step(theta - eps * dtheta, m - eps * dm, x, y, lr, mu)
        fd_theta = (tp - tm) / (2 * eps)
        fd_m = (mp - mm) / (2 * eps)
        err = max(rel_error(manual_dt, fd_theta, 1e-30), rel_error(manual_dm, fd_m, 1e-30))
        best = min(best, err)
        rows.append(
            {
                "section": "sgd",
                "check": "one_step_manual_vs_centered_fd",
                "epsilon": eps,
                "passed": err < 1e-5,
                "blocking": False,
                "relative_error": err,
                "cosine": cosine(manual_dt, fd_theta, 1e-30),
                "detail": "SGD one-step tangent finite-difference convergence point",
            }
        )
    rows.append(
        {
            "section": "sgd",
            "check": "one_step_best_error",
            "passed": best < 1e-7,
            "blocking": True,
            "relative_error": best,
            "cosine": np.nan,
            "detail": "best SGD one-step finite-difference error across eps sweep",
        }
    )

    
    z = torch.randn_like(theta)
    dtheta = torch.zeros_like(theta)
    dm = z.clone()
    th, mm = theta.clone(), m.clone()
    for _ in range(5):
        th, mm, dtheta, dm = _sgd_tangent(th, mm, dtheta, dm, x, y, lr, mu)
    plus_th, plus_m = theta.clone(), m + 1e-4 * z
    minus_th, minus_m = theta.clone(), m - 1e-4 * z
    for _ in range(5):
        plus_th, plus_m = _sgd_step(plus_th, plus_m, x, y, lr, mu)
        minus_th, minus_m = _sgd_step(minus_th, minus_m, x, y, lr, mu)
    fd = (plus_th - minus_th) / (2e-4)
    rows.append(
        {
            "section": "sgd",
            "check": "multistep_momentum_response_H5",
            "passed": rel_error(dtheta, fd, 1e-30) < 1e-7,
            "blocking": True,
            "relative_error": rel_error(dtheta, fd, 1e-30),
            "cosine": cosine(dtheta, fd, 1e-30),
            "detail": "H=5 chronological tangent response matches centered finite difference",
        }
    )

    
    b1, b2, eps_adam, adam_lr = 0.9, 0.999, 1e-8, 3e-4
    m_adam = torch.randn_like(theta) * 0.01
    v_adam = torch.rand_like(theta) * 0.01 + 1e-5
    step = 17
    for component in ("m", "rms_global"):
        q = torch.randn_like(theta)
        if component == "m":
            dm0, dv0 = q, torch.zeros_like(q)
            plus = (theta, m_adam + 1e-4 * q, v_adam)
            minus = (theta, m_adam - 1e-4 * q, v_adam)
        else:
            q = torch.ones_like(theta)
            dm0, dv0 = torch.zeros_like(q), 2 * v_adam * q
            plus = (theta, m_adam, v_adam * torch.exp(2e-4 * q))
            minus = (theta, m_adam, v_adam * torch.exp(-2e-4 * q))
        _, _, _, _, manual_dt, _, _ = _adam_tangent(
            theta,
            m_adam,
            v_adam,
            step,
            torch.zeros_like(theta),
            dm0,
            dv0,
            x,
            y,
            adam_lr,
            b1,
            b2,
            eps_adam,
        )
        plus_theta, *_ = _adam_step(*plus, step, x, y, adam_lr, b1, b2, eps_adam)
        minus_theta, *_ = _adam_step(*minus, step, x, y, adam_lr, b1, b2, eps_adam)
        fd = (plus_theta - minus_theta) / (2e-4)
        rows.append(
            {
                "section": "adam",
                "check": f"one_step_{component}_manual_vs_centered_fd",
                "passed": rel_error(manual_dt, fd, 1e-30) < 1e-6,
                "blocking": True,
                "relative_error": rel_error(manual_dt, fd, 1e-30),
                "cosine": cosine(manual_dt, fd, 1e-30),
                "detail": "Adam full-coupled one-step tangent matches centered finite difference",
            }
        )
        if component != "m":
            rows.append(
                {
                    "section": "adam",
                    "check": "rms_perturbations_nonnegative",
                    "passed": bool((plus[2] >= 0).all() and (minus[2] >= 0).all()),
                    "blocking": True,
                    "relative_error": np.nan,
                    "cosine": np.nan,
                    "detail": "v0 exp(2 alpha q) preserves nonnegative second moment",
                }
            )

    report = pd.DataFrame(rows)
    atomic_write_csv(exact_optimizer_equations(), output_dir / "audit_optimizer_equations.csv")
    atomic_write_csv(report, output_dir / "preflight_numerical_audit.csv")

    failed = report[(~report["passed"].astype(bool)) & (report["blocking"].astype(bool))]
    summary = pd.DataFrame(
        [
            {
                "check": "preflight_numerical_audit",
                "passed": len(failed) == 0,
                "blocking": True,
                "detail": "all blocking numerical audit checks passed" if len(failed) == 0 else failed[["section", "check", "relative_error"]].to_dict("records"),
            },
            {
                "check": "optimizer_equations_documented",
                "passed": True,
                "blocking": False,
                "detail": "audit_optimizer_equations.csv written",
            },
        ]
    )
    atomic_write_csv(summary, output_dir / "preflight_audit_summary.csv")
    if len(failed):
        raise AssertionError(f"blocking preflight audit checks failed: {failed[['section','check','relative_error']].to_dict('records')}")
    return summary

