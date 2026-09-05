from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class LoadedModel:
    model: object
    preprocess: Callable
    image_size: int
    spec: str
    embed_dim: int


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    if ":" not in spec:
        raise ValueError(f"Model spec must be '<source>:<name>[:<weights>]', got {spec!r}")
    parts = spec.split(":")
    if parts[0] not in ("open_clip", "timm", "mock"):
        raise ValueError(f"Unknown model source {parts[0]!r}; expected 'open_clip', 'timm', or 'mock'")
    source = parts[0]
    name = parts[1] if len(parts) > 1 else ""
    tag = parts[2] if len(parts) > 2 else None
    return source, name, tag


def load_model(spec: str, pretrained: bool = True, device: str = "cpu") -> LoadedModel:
    source, name, tag = parse_spec(spec)
    if source == "open_clip":
        return _load_open_clip(name, spec, tag or "openai", device)
    if source == "timm":
        return _load_timm(name, spec, pretrained, device)
    raise ValueError(f"Source {source!r} has no downloadable weights; build it in code")


def _load_open_clip(name: str, spec: str, tag: str, device: str) -> LoadedModel:
    try:
        import open_clip
        import torch
    except ImportError as e:
        raise ImportError("open_clip-torch is required for open_clip models: pip install -e '.[open-clip]'") from e
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=tag)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    dim = getattr(model.visual, "output_dim", 512)
    size = int(getattr(model.visual, "image_size", 224))
    if isinstance(size, (tuple, list)):
        size = int(size[0])
    return LoadedModel(model=model, preprocess=preprocess, image_size=size, spec=spec, embed_dim=int(dim))


def _load_timm(name: str, spec: str, pretrained: bool, device: str) -> LoadedModel:
    try:
        import timm
        import torch
        from torchvision import transforms
    except ImportError as e:
        raise ImportError("timm + torchvision are required for timm models") from e
    model = timm.create_model(name, pretrained=pretrained)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(data_cfg["input_size"][1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
    ])
    size = int(data_cfg["input_size"][1])
    dim = int(getattr(model, "num_features", 768))
    return LoadedModel(model=model, preprocess=preprocess, image_size=size, spec=spec, embed_dim=dim)


class ActivationHook:
    def __init__(self, model: object, layer_name: str):
        import re
        import torch.nn as nn
        if "__" in layer_name or not re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*|\d+)(\.([A-Za-z_][A-Za-z0-9_]*|\d+))*", layer_name):
            raise ValueError(f"Invalid hook name {layer_name!r}")
        module = model
        for part in layer_name.split("."):
            if part.isdigit() and hasattr(module, "__getitem__"):
                module = module[int(part)]
            else:
                module = getattr(module, part)
        if not isinstance(module, nn.Module):
            raise ValueError(f"Hook target {layer_name!r} is not a module")
        self._module = module
        self.captured = None
        self._handle = None

    def __enter__(self) -> "ActivationHook":
        def _fn(_m, _i, output):
            self.captured = output[0] if isinstance(output, tuple) else output
        self._handle = self._module.register_forward_hook(_fn)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None


def encode_images_open_clip(loaded: LoadedModel, batch) -> object:
    import torch
    with torch.no_grad():
        feats = loaded.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def resolve_hook_name(spec: str, layer: int, component: str) -> str:
    if layer is None or layer < 0:
        raise ValueError("SAE declares no hook layer; cannot resolve hook point")
    source = spec.split(":")[0]
    comp = (component or "").lower()
    is_mlp = "mlp" in comp
    if source == "open_clip":
        base = f"visual.transformer.resblocks.{layer}"
        return base + ".mlp" if is_mlp else base
    if source == "timm":
        base = f"blocks.{layer}"
        return base + ".mlp" if is_mlp else base
    if source == "mock":
        return f"blocks.{layer}"
    raise ValueError(f"Cannot resolve hook point for model source {source!r}")
