from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _tuple_env(name: str, default: tuple[Any, ...], cast=str) -> tuple[Any, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(cast(x.strip()) for x in value.split(",") if x.strip())


@dataclass(frozen=True)
class GridConfig:
    
    run_name: str = "rttp_full_grid"
    root_name: str = "rttp_full_grid"
    drive_base: str = "/content/drive/MyDrive"
    use_google_drive: bool = False
    paper_material_dir: str = ""

    
    datasets: tuple[str, ...] = ("split_cifar10", "split_tinyimagenet")
    architectures: tuple[str, ...] = ("smallcnn", "resnet18")
    optimizers: tuple[str, ...] = ("heavy_ball", "adam")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    
    cifar10_dataset_id: str = "uoft-cs/cifar10"
    tinyimagenet_dataset_id: str = "zh-plus/tiny-imagenet"
    cifar10_task_a: tuple[int, ...] = (0, 1, 2, 3, 4)
    cifar10_task_b: tuple[int, ...] = (5, 6, 7, 8, 9)
    tiny_task_a: tuple[int, ...] = tuple(range(100))
    tiny_task_b: tuple[int, ...] = tuple(range(100, 200))
    cifar_train_per_class: int = 1000
    cifar_test_per_class: int = 500
    tiny_train_per_class: int = 500
    tiny_test_per_class: int = 50
    batch_size: int = 128
    num_workers: int = 2

    
    boundary_epochs_smallcnn_cifar: int = 12
    boundary_epochs_resnet18_cifar: int = 20
    boundary_epochs_smallcnn_tiny: int = 10
    boundary_epochs_resnet18_tiny: int = 15
    boundary_sgd_lr_cifar: float = 0.03
    boundary_sgd_lr_tiny: float = 0.01
    boundary_sgd_momentum: float = 0.9
    boundary_adam_lr_cifar: float = 1e-3
    boundary_adam_lr_tiny: float = 3e-4
    boundary_weight_decay: float = 0.0

    
    horizons_sgd: tuple[int, ...] = (1, 5, 10, 20, 40)
    horizons_adam: tuple[int, ...] = (1, 5, 10, 20)
    sgd_response_lrs_cifar: tuple[float, ...] = (0.01, 0.03)
    sgd_response_lrs_tiny: tuple[float, ...] = (0.003, 0.01)
    sgd_momenta: tuple[float, ...] = (0.5, 0.9)
    adam_response_lrs_cifar: tuple[float, ...] = (3e-5, 3e-4)
    adam_response_lrs_tiny: tuple[float, ...] = (1e-5, 1e-4)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    adam_memory_components: tuple[str, ...] = (
        "m",
        "rms_global",
        "rms_layerwise",
        "rms_random_layerwise",
    )

    
    n_directions: int = 4
    sgd_fd_scales: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    adam_m_fd_scales: tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    adam_rms_fd_scales: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    resnet_rms_groups: str = "block"

    
    force: bool = False
    run_preflight_audit: bool = True
    run_dataset_probe: bool = True
    run_boundary: bool = True
    run_response: bool = True
    run_frozen_baseline: bool = True
    run_summary: bool = True
    run_figures: bool = True
    dtype: str = "float32"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "GridConfig":
        smoke = os.environ.get("RTTP_FULL_GRID_SMOKE", "0") == "1"
        if smoke:
            return cls(
                run_name=os.environ.get("RTTP_RUN_NAME", "rttp_full_grid_smoke"),
                root_name=os.environ.get("RTTP_ROOT_NAME", "rttp_full_grid_smoke"),
                datasets=_tuple_env("RTTP_DATASETS", ("split_cifar10",)),
                architectures=_tuple_env("RTTP_ARCHITECTURES", ("smallcnn",)),
                optimizers=_tuple_env("RTTP_OPTIMIZERS", ("heavy_ball", "adam")),
                seeds=_tuple_env("RTTP_SEEDS", (0,), int),
                cifar_train_per_class=50,
                cifar_test_per_class=50,
                tiny_train_per_class=20,
                tiny_test_per_class=10,
                batch_size=64,
                boundary_epochs_smallcnn_cifar=1,
                boundary_epochs_resnet18_cifar=1,
                boundary_epochs_smallcnn_tiny=1,
                boundary_epochs_resnet18_tiny=1,
                horizons_sgd=(1, 3),
                horizons_adam=(1, 3),
                sgd_response_lrs_cifar=(0.01,),
                sgd_response_lrs_tiny=(0.003,),
                sgd_momenta=(0.9,),
                adam_response_lrs_cifar=(3e-5,),
                adam_response_lrs_tiny=(1e-5,),
                adam_memory_components=("m", "rms_global"),
                n_directions=1,
                num_workers=0,
                run_figures=False,
                run_frozen_baseline=True,
                force=os.environ.get("RTTP_FORCE", "0") == "1",
            )
        return cls(
            run_name=os.environ.get("RTTP_RUN_NAME", "rttp_full_grid"),
            root_name=os.environ.get("RTTP_ROOT_NAME", "rttp_full_grid"),
            datasets=_tuple_env("RTTP_DATASETS", ("split_cifar10", "split_tinyimagenet")),
            architectures=_tuple_env("RTTP_ARCHITECTURES", ("smallcnn", "resnet18")),
            optimizers=_tuple_env("RTTP_OPTIMIZERS", ("heavy_ball", "adam")),
            seeds=_tuple_env("RTTP_SEEDS", (0, 1, 2, 3, 4), int),
            run_frozen_baseline=os.environ.get("RTTP_RUN_FROZEN_BASELINE", "1") != "0",
            force=os.environ.get("RTTP_FORCE", "0") == "1",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "GridConfig":
        payload = json.loads(Path(path).read_text())
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=list))
