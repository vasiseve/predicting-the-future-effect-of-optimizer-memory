from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchvision.models import resnet18


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def build_model(architecture: str, n_classes: int) -> nn.Module:
    if architecture == "smallcnn":
        return SmallCNN(n_classes)
    if architecture == "resnet18":
        model = resnet18(weights=None, num_classes=n_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(architecture)


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def named_buffers_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: b.detach().cpu().clone() for name, b in model.named_buffers()}


def load_buffers(model: nn.Module, buffers: Mapping[str, torch.Tensor]) -> None:
    own = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in buffers.items():
            if name in own:
                own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))


class FunctionalModel:
    def __init__(self, model: nn.Module):
        self.model = model
        self.param_items = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.param_names = [n for n, _ in self.param_items]
        self.param_specs = [(p.shape, p.numel()) for _, p in self.param_items]
        self.num_params = int(sum(n for _, n in self.param_specs))

    def vector_to_param_dict(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        offset = 0
        for name, (shape, n) in zip(self.param_names, self.param_specs):
            out[name] = theta[offset : offset + n].view(shape)
            offset += n
        if offset != theta.numel():
            raise ValueError("parameter vector size mismatch")
        return out

    def buffer_dict(self) -> dict[str, torch.Tensor]:
        return {name: b for name, b in self.model.named_buffers()}

    def loss(self, theta: torch.Tensor, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        logits = functional_call(self.model, (self.vector_to_param_dict(theta), self.buffer_dict()), (xb,))
        return F.cross_entropy(logits, yb)


def parameter_group_slices(architecture: str, names: list[str], specs, resnet_groups: str) -> dict[str, slice]:
    individual = {}
    offset = 0
    for name, (_, n) in zip(names, specs):
        individual[name] = slice(offset, offset + n)
        offset += n
    if architecture != "resnet18" or resnet_groups != "block":
        return individual
    grouped: dict[str, list[int]] = {}
    for name, sl in individual.items():
        group = name.split(".")[0]
        if name.startswith("layer1."):
            group = "layer1"
        elif name.startswith("layer2."):
            group = "layer2"
        elif name.startswith("layer3."):
            group = "layer3"
        elif name.startswith("layer4."):
            group = "layer4"
        grouped.setdefault(group, []).extend(range(sl.start, sl.stop))
    return {k: slice(min(v), max(v) + 1) for k, v in grouped.items()}

