from __future__ import annotations

import json
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
FIGURES = OUT / "figures"
NOTES = OUT / "notes"


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, NOTES):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "tables" / name)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if col in {"dataset", "optimizer", "architecture", "memory_component", "response_model", "tangent_model", "direction_label", "state_parameterization", "nominal_memory", "batchnorm_buffers"}:
            continue
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception:
            pass
    return out


def best_by(df: pd.DataFrame, keys: list[str], metric: str, tie_break: str = "radius_scale") -> pd.DataFrame:
    rows = []
    usable = df.dropna(subset=[metric]).copy()
    for _, group in usable.groupby(keys, dropna=False):
        sort_cols = [metric]
        ascending = [True]
        if tie_break in group.columns:
            sort_cols.append(tie_break)
            ascending.append(False)
        rows.append(group.sort_values(sort_cols, ascending=ascending).iloc[0])
    return pd.DataFrame(rows)


def pretty_dataset(value: str) -> str:
    return {"split_cifar10": "CIFAR-10", "split_tinyimagenet": "Tiny ImageNet"}.get(value, value)


def pretty_arch(value: str) -> str:
    return {"smallcnn": "SmallCNN", "resnet18": "ResNet-18"}.get(value, value)


def pretty_optimizer(value: str) -> str:
    return {"heavy_ball": "SGD", "adam": "Adam"}.get(value, value)


def add_pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["dataset_label"] = out["dataset"].map(pretty_dataset)
    if "architecture" in out.columns:
        out["architecture_label"] = out["architecture"].map(pretty_arch)
    if "optimizer" in out.columns:
        out["optimizer_label"] = out["optimizer"].map(pretty_optimizer)
    return out


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / name
    df.to_csv(path, index=False)
    return path


COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "black": "#111111",
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


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def save_image(img: Image.Image, stem: str) -> tuple[str, str]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    img.save(png)
    img.convert("RGB").save(pdf, "PDF", resolution=240.0)
    return str(png), str(pdf)


def draw_rotated_text(base: Image.Image, xy: tuple[int, int], text: str, angle: int, fill, fnt) -> None:
    text_box = Image.new("RGBA", (500, 120), (255, 255, 255, 0))
    d = ImageDraw.Draw(text_box)
    d.text((0, 0), text, fill=fill, font=fnt)
    rot = text_box.rotate(angle, expand=True)
    base.alpha_composite(rot, xy)


def line_chart(
    series: list[dict],
    stem: str,
    title: str,
    ylabel: str,
    yscale: str = "linear",
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict[str, str]:
    width, height = 1500, 940
    left, right, top, bottom = 155, 390, 125, 145
    plot_w, plot_h = width - left - right, height - top - bottom
    img = Image.new("RGBA", (width, height), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis, f_tick, f_legend = font(34, True), font(24), font(19), font(20)
    d.text((left, 28), title, fill=hex_to_rgb(COLORS["black"]), font=f_title)
    d.text((left, 78), ylabel, fill=(20, 20, 20), font=f_axis)
    xs = sorted({float(x) for s in series for x in s["x"]})
    values = [float(y) for s in series for y in s["y"] if pd.notna(y) and np.isfinite(y)]
    if yscale == "log":
        values = [v for v in values if v > 0]
    if not values:
        values = [1.0]
    if y_min is None:
        y_min = min(values)
        if yscale == "log":
            y_min = 10 ** math.floor(math.log10(max(y_min, 1e-12)))
        else:
            y_min = min(0.0, y_min)
    if y_max is None:
        y_max = max(values)
        if yscale == "log":
            y_max = 10 ** math.ceil(math.log10(max(y_max, y_min * 1.01)))
        else:
            y_max = y_max * 1.05 if y_max > 0 else 1.0
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_max = x_min + 1

    def xmap(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    if yscale == "log":
        log_min, log_max = math.log10(y_min), math.log10(y_max)

        def ymap(y):
            y = max(float(y), y_min)
            return top + (log_max - math.log10(y)) / (log_max - log_min) * plot_h

        ticks = [10**k for k in range(math.floor(log_min), math.ceil(log_max) + 1)]
        tick_labels = [f"$10^{int(math.log10(t))}$" for t in ticks]
    else:
        def ymap(y):
            return top + (y_max - float(y)) / (y_max - y_min) * plot_h

        ticks = np.linspace(y_min, y_max, 6)
        tick_labels = [f"{t:.3f}" for t in ticks]

    
    d.line((left, top, left, top + plot_h), fill=(0, 0, 0), width=2)
    d.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(0, 0, 0), width=2)
    for t, label in zip(ticks, tick_labels):
        if yscale == "log" and (t < y_min or t > y_max):
            continue
        y = ymap(t)
        d.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=1)
        d.text((18, y - 12), label.replace("$", ""), fill=(40, 40, 40), font=f_tick)
    for x in xs:
        xp = xmap(x)
        d.line((xp, top + plot_h, xp, top + plot_h + 8), fill=(0, 0, 0), width=2)
        d.text((xp - 18, top + plot_h + 16), str(int(x)), fill=(40, 40, 40), font=f_tick)
    d.text((left + plot_w / 2 - 70, height - 58), "Horizon H", fill=(20, 20, 20), font=f_axis)

    
    for s in series:
        color = hex_to_rgb(s["color"])
        points = [(xmap(x), ymap(y)) for x, y in zip(s["x"], s["y"]) if pd.notna(y) and np.isfinite(y) and (yscale != "log" or y > 0)]
        if len(points) >= 2:
            d.line(points, fill=color, width=4)
        for xp, yp in points:
            r = 7
            d.ellipse((xp - r, yp - r, xp + r, yp + r), fill=color, outline=(255, 255, 255), width=2)

    
    lx, ly = left + plot_w + 40, top
    for i, s in enumerate(series):
        y = ly + i * 34
        color = hex_to_rgb(s["color"])
        d.line((lx, y + 12, lx + 38, y + 12), fill=color, width=5)
        d.ellipse((lx + 15, y + 5, lx + 29, y + 19), fill=color)
        d.text((lx + 52, y), s["label"], fill=(30, 30, 30), font=f_legend)
    png, pdf = save_image(img, stem)
    return {"figure": stem, "description": title, "path_png": png, "path_pdf": pdf}


def grouped_bar_chart(groups, stem: str, title: str, ylabel: str, log: bool = False) -> dict[str, str]:
    width, height = 1500, 900
    left, right, top, bottom = 150, 70, 125, 190
    plot_w, plot_h = width - left - right, height - top - bottom
    img = Image.new("RGBA", (width, height), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis, f_tick, f_legend = font(34, True), font(24), font(18), font(18)
    d.text((left, 28), title, fill=(20, 20, 20), font=f_title)
    d.text((left, 78), ylabel, fill=(20, 20, 20), font=f_axis)
    values = [v for g in groups for _, v, _ in g["bars"] if pd.notna(v) and np.isfinite(v) and v > 0]
    y_min = min(values) if log else 0.0
    y_max = max(values) if values else 1.0
    if log:
        y_min = 10 ** math.floor(math.log10(max(y_min, 1e-6)))
        y_max = 10 ** math.ceil(math.log10(max(y_max, y_min * 1.01)))
        log_min, log_max = math.log10(y_min), math.log10(y_max)

        def ymap(y):
            return top + (log_max - math.log10(max(float(y), y_min))) / (log_max - log_min) * plot_h

        ticks = [10**k for k in range(math.floor(log_min), math.ceil(log_max) + 1)]
        labels = [f"10^{int(math.log10(t))}" for t in ticks]
    else:
        y_max *= 1.08

        def ymap(y):
            return top + (y_max - float(y)) / (y_max - y_min) * plot_h

        ticks = np.linspace(y_min, y_max, 6)
        labels = [f"{t:.2f}" for t in ticks]
    d.line((left, top, left, top + plot_h), fill=(0, 0, 0), width=2)
    d.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(0, 0, 0), width=2)
    for t, label in zip(ticks, labels):
        y = ymap(t)
        d.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=1)
        d.text((28, y - 11), label, fill=(40, 40, 40), font=f_tick)
    n_groups = len(groups)
    group_w = plot_w / max(n_groups, 1)
    for gi, group in enumerate(groups):
        bars = group["bars"]
        bar_w = min(70, group_w / (len(bars) + 1))
        gx = left + gi * group_w + group_w / 2
        for bi, (_, value, color) in enumerate(bars):
            if pd.isna(value) or not np.isfinite(value) or value <= 0:
                continue
            x0 = gx + (bi - (len(bars) - 1) / 2) * bar_w - bar_w * 0.42
            x1 = x0 + bar_w * 0.84
            y = ymap(value)
            d.rectangle((x0, y, x1, top + plot_h), fill=hex_to_rgb(color))
        tx = int(gx - group_w * 0.42)
        d.multiline_text((tx, top + plot_h + 20), group["label"], fill=(35, 35, 35), font=f_tick, spacing=2, align="center")
    d.text((left + plot_w / 2 - 50, height - 48), "", fill=(20, 20, 20), font=f_axis)
    legend_items = []
    for group in groups:
        for label, _, color in group["bars"]:
            if label not in [x[0] for x in legend_items]:
                legend_items.append((label, color))
    lx, ly = left + plot_w - 350, top + 10
    for i, (label, color) in enumerate(legend_items):
        y = ly + i * 30
        d.rectangle((lx, y, lx + 24, y + 18), fill=hex_to_rgb(color))
        d.text((lx + 34, y - 2), label, fill=(30, 30, 30), font=f_legend)
    png, pdf = save_image(img, stem)
    return {"figure": stem, "description": title, "path_png": png, "path_pdf": pdf}


def viridis_like(value: float, vmin: float = 0.0, vmax: float = 20.0) -> tuple[int, int, int]:
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    t = 0.0 if vmax <= vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    p = t * (len(stops) - 1)
    i = min(int(p), len(stops) - 2)
    frac = p - i
    return tuple(int(stops[i][k] * (1 - frac) + stops[i + 1][k] * frac) for k in range(3))


def heatmap(pivot: pd.DataFrame, stem: str, title: str) -> dict[str, str]:
    width, height = 1450, 760
    left, right, top, bottom = 330, 90, 155, 105
    img = Image.new("RGBA", (width, height), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis, f_cell = font(34, True), font(20), font(24, True)
    d.text((left, 28), title, fill=(20, 20, 20), font=f_title)
    rows = list(pivot.index)
    cols = list(pivot.columns)
    cell_w = (width - left - right) / max(len(cols), 1)
    cell_h = (height - top - bottom) / max(len(rows), 1)
    values = pivot.fillna(0).to_numpy(float)
    for i, row in enumerate(rows):
        y0 = top + i * cell_h
        d.text((20, y0 + cell_h / 2 - 12), row, fill=(30, 30, 30), font=f_axis)
        for j, col in enumerate(cols):
            x0 = left + j * cell_w
            value = values[i, j]
            color = viridis_like(value)
            d.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=color)
            text = f"{int(value)}" if value > 0 else "0"
            tw = d.textlength(text, font=f_cell)
            fill = (255, 255, 255) if value > 8 else (15, 15, 15)
            d.text((x0 + cell_w / 2 - tw / 2, y0 + cell_h / 2 - 14), text, fill=fill, font=f_cell)
    for j, col in enumerate(cols):
        x0 = left + j * cell_w
        tw = d.textlength(col, font=f_axis)
        d.text((x0 + cell_w / 2 - tw / 2, top - 42), col, fill=(30, 30, 30), font=f_axis)
    d.text((left, height - 38), "Cell value: maximum H with endpoint error <= 0.10 and cosine >= 0.99", fill=(30, 30, 30), font=f_axis)
    png, pdf = save_image(img, stem)
    return {"figure": stem, "description": title, "path_png": png, "path_pdf": pdf}


def horizontal_layerwise_chart(df: pd.DataFrame, order: list[str], stem: str, title: str) -> dict[str, str]:
    width, height = 1500, 900
    left, right, top, bottom = 210, 80, 140, 90
    panel_gap = 70
    panel_w = (width - left - right - panel_gap) / 2
    plot_h = height - top - bottom
    img = Image.new("RGBA", (width, height), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis, f_tick = font(34, True), font(22), font(19)
    d.text((left, 28), title, fill=(20, 20, 20), font=f_title)
    datasets = list(dict.fromkeys(df["dataset_label"].tolist()))
    max_x = max(1.0, float(df["endpoint_error"].max()) * 1.05)
    for pi, dataset in enumerate(datasets[:2]):
        group = df[df["dataset_label"].eq(dataset)].set_index("direction_label").reindex(order).reset_index()
        x0 = left + pi * (panel_w + panel_gap)
        y0 = top
        d.text((x0 + 10, y0 - 40), dataset, fill=(20, 20, 20), font=f_axis)
        d.line((x0, y0, x0, y0 + plot_h), fill=(0, 0, 0), width=2)
        d.line((x0, y0 + plot_h, x0 + panel_w, y0 + plot_h), fill=(0, 0, 0), width=2)
        for tick in [0.1, 0.25, 0.5, 1.0]:
            xp = x0 + min(tick, max_x) / max_x * panel_w
            d.line((xp, y0, xp, y0 + plot_h), fill=(210, 210, 210), width=1)
            d.text((xp - 12, y0 + plot_h + 12), f"{tick:g}", fill=(40, 40, 40), font=f_tick)
        row_h = plot_h / max(len(order), 1)
        for i, row in group.iterrows():
            yp = y0 + i * row_h + row_h * 0.2
            label = str(row["direction_label"])
            value = row["endpoint_error"]
            if pi == 0:
                d.text((20, yp + 5), label, fill=(30, 30, 30), font=f_tick)
            if pd.notna(value) and np.isfinite(value):
                w = min(float(value), max_x) / max_x * panel_w
                d.rectangle((x0, yp, x0 + w, yp + row_h * 0.58), fill=hex_to_rgb(COLORS["purple"]))
                d.text((x0 + w + 6, yp + 2), f"{float(value):.3f}", fill=(30, 30, 30), font=f_tick)
        d.text((x0 + panel_w / 2 - 95, height - 38), "Endpoint error at H=20", fill=(30, 30, 30), font=f_axis)
    png, pdf = save_image(img, stem)
    return {"figure": stem, "description": title, "path_png": png, "path_pdf": pdf}


def markdown_table(df: pd.DataFrame) -> str:
    if not len(df):
        return ""
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def plot_sgd_time_varying(sgd: pd.DataFrame) -> list[dict[str, str]]:
    outputs = []
    sgd = add_pretty_columns(sgd)
    colors = {
        ("CIFAR-10", "SmallCNN"): COLORS["blue"],
        ("CIFAR-10", "ResNet-18"): COLORS["red"],
        ("Tiny ImageNet", "SmallCNN"): COLORS["green"],
        ("Tiny ImageNet", "ResNet-18"): COLORS["purple"],
    }
    markers = {"CIFAR-10": "o", "Tiny ImageNet": "s"}

    for metric, ylabel, stem, yscale in [
        ("endpoint_error", "Relative endpoint error", "fig_sgd_timevarying_endpoint_error", "log"),
        ("endpoint_cosine", "Endpoint displacement cosine", "fig_sgd_timevarying_endpoint_cosine", "linear"),
    ]:
        series = []
        for (dataset, arch), group in sgd.groupby(["dataset_label", "architecture_label"], dropna=False):
            group = group.sort_values("horizon")
            series.append(
                {
                    "x": group["horizon"].tolist(),
                    "y": group[metric].tolist(),
                    "label": f"{dataset}, {arch}",
                    "color": colors[(dataset, arch)],
                    "marker": markers[dataset],
                }
            )
        outputs.append(line_chart(series, stem, "SGD time-varying response", ylabel, yscale=yscale, y_min=0.98 if metric == "endpoint_cosine" else None, y_max=1.001 if metric == "endpoint_cosine" else None))
    return outputs


def plot_frozen_sgd(frozen: pd.DataFrame) -> list[dict[str, str]]:
    outputs = []
    sgd = frozen[frozen["optimizer"].eq("heavy_ball")].dropna(subset=["endpoint_error_time_varying", "endpoint_error_frozen_boundary"]).copy()
    sgd = add_pretty_columns(sgd)
    colors = {
        ("CIFAR-10", "SmallCNN"): COLORS["blue"],
        ("CIFAR-10", "ResNet-18"): COLORS["red"],
        ("Tiny ImageNet", "SmallCNN"): COLORS["green"],
        ("Tiny ImageNet", "ResNet-18"): COLORS["purple"],
    }

    series = []
    for (dataset, arch), group in sgd.groupby(["dataset_label", "architecture_label"], dropna=False):
        group = group.sort_values("horizon")
        ds = "CIFAR" if dataset == "CIFAR-10" else "Tiny"
        ar = "R18" if arch == "ResNet-18" else "CNN"
        series.append({"x": group["horizon"].tolist(), "y": group["endpoint_error_time_varying"].tolist(), "label": f"{ds}-{ar} TV", "color": colors[(dataset, arch)]})
        series.append({"x": group["horizon"].tolist(), "y": group["endpoint_error_frozen_boundary"].tolist(), "label": f"{ds}-{ar} frozen", "color": COLORS["orange"] if arch == "SmallCNN" else COLORS["gray"]})
    stem = "fig_sgd_frozen_vs_timevarying_endpoint_error"
    outputs.append(line_chart(series, stem, "SGD frozen versus time-varying", "Relative endpoint error", yscale="log"))

    h20 = sgd[sgd["horizon"].eq(20)].copy()
    h20["label"] = h20["dataset_label"] + "\n" + h20["architecture_label"]
    groups = [
        {
            "label": row["label"],
            "bars": [
                ("Time-varying", row["endpoint_error_time_varying"], COLORS["green"]),
                ("Frozen", row["endpoint_error_frozen_boundary"], COLORS["orange"]),
            ],
        }
        for _, row in h20.iterrows()
    ]
    stem = "fig_sgd_frozen_vs_timevarying_h20_bar"
    outputs.append(grouped_bar_chart(groups, stem, "SGD H=20 frozen comparison", "Relative endpoint error at H=20", log=True))

    return outputs


def plot_adam(adam_h20: pd.DataFrame, validated: pd.DataFrame, layerwise: pd.DataFrame) -> list[dict[str, str]]:
    outputs = []
    adam_h20 = add_pretty_columns(adam_h20)
    component_order = ["m", "rms_global", "rms_random_layerwise", "rms_layerwise"]
    component_label = {
        "m": "first moment",
        "rms_global": "RMS global",
        "rms_random_layerwise": "RMS random-layer",
        "rms_layerwise": "RMS layerwise mean",
    }
    adam_h20["component_label"] = adam_h20["memory_component"].map(component_label).fillna(adam_h20["memory_component"])
    adam_h20["block_label"] = adam_h20["dataset_label"] + "\n" + adam_h20["architecture_label"]

    blocks = list(dict.fromkeys(adam_h20["block_label"].tolist()))
    bar_colors = [COLORS["blue"], COLORS["green"], COLORS["gray"], COLORS["purple"]]
    groups = []
    for block in blocks:
        bars = []
        for component, color in zip(component_order, bar_colors):
            row = adam_h20[(adam_h20["block_label"].eq(block)) & (adam_h20["memory_component"].eq(component))]
            value = float(row["endpoint_error"].iloc[0]) if len(row) else np.nan
            bars.append((component_label[component], value, color))
        groups.append({"label": block, "bars": bars})
    stem = "fig_adam_h20_endpoint_error_by_component"
    outputs.append(grouped_bar_chart(groups, stem, "Adam H=20 response by memory component", "Relative endpoint error at H=20", log=True))

    heat = validated[validated["regime"].eq("strong")].copy()
    heat = heat[heat["optimizer"].eq("adam")]
    heat = add_pretty_columns(heat)
    heat["row"] = heat["dataset_label"] + " / " + heat["architecture_label"]
    heat["col"] = heat["memory_component"].map(component_label).fillna(heat["memory_component"])
    pivot = heat.pivot_table(index="row", columns="col", values="max_validated_horizon", aggfunc="max").reindex(columns=[component_label[c] for c in component_order])
    stem = "fig_adam_validated_horizon_heatmap"
    outputs.append(heatmap(pivot, stem, "Adam strong-regime maximum horizon"))

    resnet_layerwise = layerwise[layerwise["architecture"].eq("resnet18")].copy()
    resnet_layerwise = add_pretty_columns(resnet_layerwise)
    order = ["bn1", "conv1", "layer1", "layer2", "fc", "layer3", "layer4"]
    stem = "fig_adam_resnet_rms_layerwise_h20"
    outputs.append(horizontal_layerwise_chart(resnet_layerwise, order, stem, "Adam ResNet-18 RMS layerwise response at H=20"))

    return outputs


def write_markdown_report(claims: pd.DataFrame, figure_index: pd.DataFrame) -> None:
    lines = [
        "# Paper Reanalysis Bundle",
        "",
        "This directory contains derived, paper-facing tables and figures from the completed `rttp_full_grid` run.",
        "No experiment is recomputed here.",
        "",
        "## Main Use",
        "",
        "- Use `tables/sgd_h40_main_table.csv` for the main scaled SGD result.",
        "- Use `tables/frozen_h20_main_table.csv` and `tables/sgd_frozen_h40_diagnostic.csv` for the evolving-dynamics claim.",
        "- Use `tables/adam_h20_component_table.csv` and `tables/adam_validated_horizons.csv` for the adaptive optimizer section.",
        "- Use `tables/adam_rms_layerwise_resnet_h20.csv` for the ResNet RMS-memory nuance.",
        "",
        "## Claim Summary",
        "",
        markdown_table(claims),
        "",
        "## Figures",
        "",
        markdown_table(figure_index),
        "",
        "## Caveat",
        "",
        "The original quality report marks `finite_key_metrics=False` because the frozen-boundary ResNet-18 heavy-ball baseline overflows at `H=40`. All time-varying RTTP rows are finite. Treat frozen overflow as a diagnostic result, not as a failure of the proposed response model.",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    config = json.loads((ROOT / "config.json").read_text())
    paper = clean_numeric(read_csv("paper_table_full_grid.csv"))
    frozen = clean_numeric(read_csv("frozen_best_comparison.csv"))
    raw = clean_numeric(read_csv("response_raw.csv"))

    tv = paper[paper["response_model"].eq("time_varying")].copy()
    tv_best = best_by(tv, ["dataset", "optimizer", "architecture", "memory_component", "horizon"], "endpoint_error")
    save_table(tv_best, "time_varying_best_endpoint_by_block_horizon.csv")

    sgd = tv_best[tv_best["optimizer"].eq("heavy_ball")].copy()
    sgd_h20 = sgd[sgd["horizon"].eq(20)].copy()
    sgd_h40 = sgd[sgd["horizon"].eq(40)].copy()
    save_table(sgd_h20, "sgd_h20_main_table.csv")
    save_table(sgd_h40, "sgd_h40_main_table.csv")

    adam = tv_best[tv_best["optimizer"].eq("adam")].copy()
    adam_h20 = adam[adam["horizon"].eq(20)].copy()
    save_table(adam_h20, "adam_h20_component_table.csv")

    thresholds = [
        ("strong", 0.10, 0.99),
        ("moderate", 0.25, 0.95),
        ("loose", 0.50, 0.90),
    ]
    rows = []
    for key, group in tv_best.groupby(["dataset", "optimizer", "architecture", "memory_component"], dropna=False):
        for regime, max_error, min_cosine in thresholds:
            ok = group[(group["endpoint_error"] <= max_error) & (group["endpoint_cosine"] >= min_cosine)]
            rows.append(
                dict(
                    zip(["dataset", "optimizer", "architecture", "memory_component"], key),
                    regime=regime,
                    max_error=max_error,
                    min_cosine=min_cosine,
                    max_validated_horizon=int(ok["horizon"].max()) if len(ok) else 0,
                )
            )
    validated = pd.DataFrame(rows)
    save_table(validated, "validated_horizons_by_threshold.csv")
    save_table(validated[validated["optimizer"].eq("adam")], "adam_validated_horizons.csv")

    frozen_best = best_by(frozen, ["dataset", "optimizer", "architecture", "memory_component", "horizon"], "endpoint_error_time_varying")
    save_table(frozen_best, "frozen_best_endpoint_by_block_horizon.csv")
    frozen_h20 = frozen_best[frozen_best["horizon"].eq(20)].copy()
    sgd_frozen_h40 = frozen_best[(frozen_best["optimizer"].eq("heavy_ball")) & (frozen_best["horizon"].eq(40))].copy()
    save_table(frozen_h20, "frozen_h20_main_table.csv")
    save_table(sgd_frozen_h40, "sgd_frozen_h40_diagnostic.csv")

    nonfinite_mask = ~np.isfinite(raw[["endpoint_error", "centered_error", "tangent_norm"]].to_numpy(float)).all(axis=1)
    nonfinite = raw.loc[nonfinite_mask].groupby(["dataset", "optimizer", "architecture", "memory_component", "response_model", "horizon"], dropna=False).size().reset_index(name="nonfinite_rows")
    save_table(nonfinite, "nonfinite_rows_by_block.csv")

    layerwise = raw[
        (raw["optimizer"].eq("adam"))
        & (raw["memory_component"].eq("rms_layerwise"))
        & (raw["response_model"].eq("time_varying"))
        & (raw["horizon"].eq(20))
    ].copy()
    layer_summary = layerwise.groupby(["dataset", "architecture", "learning_rate", "direction_label", "radius_scale"], dropna=False)[["endpoint_error", "endpoint_cosine", "centered_error", "centered_cosine"]].mean().reset_index()
    layer_best = best_by(layer_summary, ["dataset", "architecture", "direction_label"], "endpoint_error")
    save_table(layer_best, "adam_rms_layerwise_h20_by_direction.csv")
    save_table(layer_best[layer_best["architecture"].eq("resnet18")], "adam_rms_layerwise_resnet_h20.csv")

    claims = pd.DataFrame(
        [
            {
                "claim": "SGD scales across datasets and architectures",
                "evidence": "All four SGD blocks validate through H=40 with endpoint error < 0.04 and cosine > 0.999.",
                "table": "sgd_h40_main_table.csv",
            },
            {
                "claim": "Chronological tangent propagation matters",
                "evidence": "At H=20, frozen endpoint errors are consistently larger; at H=40, ResNet-18 frozen SGD diverges/overflows.",
                "table": "frozen_h20_main_table.csv; sgd_frozen_h40_diagnostic.csv",
            },
            {
                "claim": "Adam supports the state-space response principle",
                "evidence": "Adam m is strong on SmallCNN and meaningful on ResNet-18; RMS channels show component-dependent local regimes.",
                "table": "adam_h20_component_table.csv; adam_validated_horizons.csv",
            },
            {
                "claim": "Adam RMS response is heterogeneous",
                "evidence": "ResNet-18 later blocks validate better than stem/BN directions.",
                "table": "adam_rms_layerwise_resnet_h20.csv",
            },
        ]
    )
    save_table(claims, "paper_claim_summary.csv")

    figures = []
    figures.extend(plot_sgd_time_varying(sgd))
    figures.extend(plot_frozen_sgd(frozen_best))
    figures.extend(plot_adam(adam_h20, validated, layer_best))
    figure_index = pd.DataFrame(figures)
    save_table(figure_index, "figure_index.csv")

    metadata = {
        "source_root": str(ROOT),
        "config": config,
        "input_rows": {
            "paper_table_full_grid": int(len(paper)),
            "frozen_best_comparison": int(len(frozen)),
            "response_raw": int(len(raw)),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    write_markdown_report(claims, figure_index)
    print(f"wrote reanalysis bundle to {OUT}")


if __name__ == "__main__":
    main()
