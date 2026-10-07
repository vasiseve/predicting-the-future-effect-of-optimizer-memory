from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


import argparse
parser = argparse.ArgumentParser(description="Rebuild paper-facing analysis from a full-grid output root.")
parser.add_argument("--root", type=Path, default=Path("reference_results/full_grid"))
ROOT = parser.parse_args().root.resolve()
OUT = ROOT / "reanalysis"
TABLES = OUT / "tables"
FIGURES = OUT / "figures_ci"


COLORS = {
    "time_varying": "#1f77b4",
    "frozen_boundary": "#ff7f0e",
    "grid": "#dddddd",
    "text": "#161616",
}


def font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    return (*rgb(hex_color), alpha)


def save_image(img: Image.Image, stem: str) -> tuple[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    img.save(png)
    img.convert("RGB").save(pdf, "PDF", resolution=240.0)
    return str(png), str(pdf)


def pretty_dataset(value: str) -> str:
    return {"split_cifar10": "CIFAR-10", "split_tinyimagenet": "Tiny ImageNet"}.get(value, value)


def pretty_arch(value: str) -> str:
    return {"smallcnn": "SmallCNN", "resnet18": "ResNet-18"}.get(value, value)


def pretty_opt(value: str) -> str:
    return {"heavy_ball": "SGD", "adam": "Adam"}.get(value, value)


def component_label(value: str, momentum: float | None = None) -> str:
    labels = {
        "momentum": f"momentum, mu={momentum:g}" if momentum is not None and np.isfinite(momentum) else "momentum",
        "m": "first moment",
        "rms_global": "RMS global",
        "rms_layerwise": "RMS layerwise mean",
        "rms_random_layerwise": "RMS random-layer",
    }
    return labels.get(value, value)


def tcrit(n: int) -> float:
    table = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return table.get(int(n), 1.96)


def ci_summary(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(x) == 0:
        return pd.Series({"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0})
    mean = float(x.mean())
    if len(x) == 1:
        return pd.Series({"mean": mean, "lo": np.nan, "hi": np.nan, "n": 1})
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    delta = tcrit(len(x)) * se
    return pd.Series({"mean": mean, "lo": mean - delta, "hi": mean + delta, "n": len(x)})


def operating_points(seed_level: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, optimizer, architecture, memory_component), group in best.groupby(["dataset", "optimizer", "architecture", "memory_component"], dropna=False):
        terminal_horizon = int(group["horizon"].max())
        row = group[group["horizon"].eq(terminal_horizon)].sort_values(["endpoint_error", "radius_scale"], ascending=[True, False]).iloc[0].to_dict()
        rows.append(
            {
                "dataset": dataset,
                "optimizer": optimizer,
                "architecture": architecture,
                "memory_component": memory_component,
                "terminal_horizon": terminal_horizon,
                "learning_rate": row["learning_rate"],
                "radius_scale": row["radius_scale"],
                "momentum": row.get("momentum", np.nan),
                "beta1": row.get("beta1", np.nan),
                "beta2": row.get("beta2", np.nan),
                "tangent_model": row.get("tangent_model", np.nan),
            }
        )
    return pd.DataFrame(rows)


def match_op(df: pd.DataFrame, op: pd.Series) -> pd.DataFrame:
    mask = (
        df["dataset"].eq(op["dataset"])
        & df["optimizer"].eq(op["optimizer"])
        & df["architecture"].eq(op["architecture"])
        & df["memory_component"].eq(op["memory_component"])
        & np.isclose(df["learning_rate"].astype(float), float(op["learning_rate"]), rtol=0, atol=1e-14)
        & np.isclose(df["radius_scale"].astype(float), float(op["radius_scale"]), rtol=0, atol=1e-14)
    )
    if op["optimizer"] == "heavy_ball":
        mask &= np.isclose(df["momentum"].astype(float), float(op["momentum"]), rtol=0, atol=1e-14)
    return df[mask].copy()


def build_curve_table(seed_level: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, op in ops.iterrows():
        sub = match_op(seed_level, op)
        if not len(sub):
            continue
        group_cols = ["dataset", "optimizer", "architecture", "memory_component", "learning_rate", "radius_scale", "momentum", "response_model", "horizon"]
        for keys, group in sub.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys))
            row.update(ci_summary(group["endpoint_error"]).to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def draw_dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill, width: int = 4, dash: int = 12, gap: int = 8) -> None:
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        t = 0.0
        while t < length:
            t2 = min(t + dash, length)
            draw.line((x0 + ux * t, y0 + uy * t, x0 + ux * t2, y0 + uy * t2), fill=fill, width=width)
            t += dash + gap


def plot_panel(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, panel: pd.DataFrame, title: str, show_ylabel: bool) -> None:
    f_panel, f_tick, f_small = font(22, True), font(16), font(16)
    draw.text((x0, y0 - 34), title, fill=rgb(COLORS["text"]), font=f_panel)
    values = []
    for col in ["mean", "lo", "hi"]:
        vals = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tolist()
        values.extend([v for v in vals if v > 0])
    y_min = 10 ** math.floor(math.log10(max(min(values), 1e-12))) if values else 1e-5
    y_max = 10 ** math.ceil(math.log10(max(values))) if values else 1.0
    if y_max <= y_min:
        y_max = y_min * 10
    horizons = sorted(panel["horizon"].dropna().unique())
    x_min, x_max = min(horizons), max(horizons)
    if x_min == x_max:
        x_max = x_min + 1
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def xm(x):
        return x0 + (float(x) - x_min) / (x_max - x_min) * w

    def ym(y):
        y = max(float(y), y_min)
        return y0 + (log_max - math.log10(y)) / (log_max - log_min) * h

    draw.line((x0, y0, x0, y0 + h), fill=(0, 0, 0), width=2)
    draw.line((x0, y0 + h, x0 + w, y0 + h), fill=(0, 0, 0), width=2)
    for k in range(math.floor(log_min), math.ceil(log_max) + 1):
        tick = 10**k
        yy = ym(tick)
        draw.line((x0, yy, x0 + w, yy), fill=rgb(COLORS["grid"]), width=1)
        if show_ylabel:
            draw.text((x0 - 64, yy - 10), f"10^{k}", fill=(40, 40, 40), font=f_tick)
    for horizon in horizons:
        xx = xm(horizon)
        draw.line((xx, y0 + h, xx, y0 + h + 6), fill=(0, 0, 0), width=2)
        draw.text((xx - 10, y0 + h + 12), str(int(horizon)), fill=(40, 40, 40), font=f_tick)

    for model in ["time_varying", "frozen_boundary"]:
        g = panel[panel["response_model"].eq(model)].sort_values("horizon")
        if not len(g):
            continue
        color = COLORS[model]
        points = [(xm(r.horizon), ym(r.mean)) for r in g.itertuples() if pd.notna(r.mean) and np.isfinite(r.mean) and r.mean > 0]
        lo = []
        hi = []
        for r in g.itertuples():
            if pd.notna(r.lo) and pd.notna(r.hi) and np.isfinite(r.lo) and np.isfinite(r.hi) and pd.notna(r.mean) and r.mean > 0:
                visual_lo = max(float(r.lo), float(r.mean) / 10.0, 1e-12)
                visual_hi = max(float(r.hi), float(r.mean), visual_lo * 1.01)
                lo.append((xm(r.horizon), ym(visual_lo)))
                hi.append((xm(r.horizon), ym(visual_hi)))
        if len(lo) == len(hi) and len(lo) >= 2:
            poly = hi + list(reversed(lo))
            draw.polygon(poly, fill=rgba(color, 48))
        if len(points) >= 2:
            if model == "frozen_boundary":
                draw_dashed_line(draw, points, rgb(color), width=4)
            else:
                draw.line(points, fill=rgb(color), width=4)
        for xx, yy in points:
            draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=rgb(color), outline=(255, 255, 255), width=2)
    draw.text((x0 + w / 2 - 40, y0 + h + 44), "Horizon H", fill=(25, 25, 25), font=f_small)


def make_figure(block: pd.DataFrame, dataset: str, optimizer: str, architecture: str) -> dict[str, str]:
    components = list(dict.fromkeys(block["memory_component"].tolist()))
    n = len(components)
    if n <= 1:
        rows, cols = 1, 1
    elif n <= 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    width, height = 1500, 900 if rows == 2 else 720
    left, right, top, bottom = 130, 310, 130, 105
    gap_x, gap_y = 95, 105
    panel_w = int((width - left - right - gap_x * (cols - 1)) / cols)
    panel_h = int((height - top - bottom - gap_y * (rows - 1)) / rows)
    img = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title = f"{pretty_dataset(dataset)} / {pretty_arch(architecture)} / {pretty_opt(optimizer)}"
    draw.text((left, 28), title, fill=rgb(COLORS["text"]), font=font(34, True))
    draw.text((left, 78), "Time-varying versus frozen endpoint error, mean with 95% CI across seeds", fill=(35, 35, 35), font=font(21))
    for i, component in enumerate(components):
        r, c = divmod(i, cols)
        x = left + c * (panel_w + gap_x)
        y = top + r * (panel_h + gap_y)
        sub = block[block["memory_component"].eq(component)]
        mom = sub["momentum"].dropna().iloc[0] if sub["momentum"].notna().any() else None
        plot_panel(draw, x, y, panel_w, panel_h, sub, component_label(component, mom), show_ylabel=(c == 0))
    lx, ly = width - right + 35, top
    f_leg = font(20)
    draw.line((lx, ly + 12, lx + 44, ly + 12), fill=rgb(COLORS["time_varying"]), width=5)
    draw.text((lx + 58, ly), "time-varying", fill=(25, 25, 25), font=f_leg)
    draw_dashed_line(draw, [(lx, ly + 50), (lx + 44, ly + 50)], rgb(COLORS["frozen_boundary"]), width=5)
    draw.text((lx + 58, ly + 38), "frozen", fill=(25, 25, 25), font=f_leg)
    stem = f"fig_frozen_ci_dataset={dataset}_optimizer={optimizer}_arch={architecture}"
    png, pdf = save_image(img, stem)
    return {"dataset": dataset, "optimizer": optimizer, "architecture": architecture, "figure": stem, "path_png": png, "path_pdf": pdf}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    seed_level = pd.read_csv(ROOT / "tables" / "seed_level_response.csv").replace([np.inf, -np.inf], np.nan)
    best = pd.read_csv(TABLES / "time_varying_best_endpoint_by_block_horizon.csv").replace([np.inf, -np.inf], np.nan)
    ops = operating_points(seed_level, best)
    curve = build_curve_table(seed_level, ops)
    ops.to_csv(TABLES / "frozen_ci_operating_points.csv", index=False)
    curve.to_csv(TABLES / "frozen_ci_curve_table.csv", index=False)
    figures = []
    for (dataset, optimizer, architecture), block in curve.groupby(["dataset", "optimizer", "architecture"], dropna=False):
        figures.append(make_figure(block, dataset, optimizer, architecture))
    pd.DataFrame(figures).to_csv(TABLES / "frozen_ci_figure_index.csv", index=False)
    readme = OUT / "notes" / "frozen_ci_figures.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# Frozen CI Figures\n\n"
        "These figures are derived from `seed_level_response.csv` and use one fixed operating point per dataset/optimizer/architecture/memory-component block. "
        "The operating point is selected by the lowest time-varying endpoint error at the terminal horizon for that optimizer/component. "
        "Curves show mean endpoint error with shaded 95% confidence intervals across seeds.\n\n"
        "Outputs:\n\n"
        "- `tables/frozen_ci_operating_points.csv`\n"
        "- `tables/frozen_ci_curve_table.csv`\n"
        "- `tables/frozen_ci_figure_index.csv`\n"
        "- `figures_ci/fig_frozen_ci_dataset=*_optimizer=*_arch=*.png/.pdf`\n"
    )
    print(f"wrote frozen CI figures to {FIGURES}")


if __name__ == "__main__":
    main()
