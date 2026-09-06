"""Image-slice MFI coordinates, independent of dataset roles and native video compression."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MFIPlan:
    targets: tuple[int, ...]
    controls: tuple[int, ...]
    output_slots: tuple[int, ...]
    output_indices: tuple[int, ...]
    origin: int


def plan_indices(targets, controls=(), *, relative=False, interpolate=False, thresholds=(2, 3), save_interpolated=False):
    targets, controls = tuple(targets), tuple(controls)
    if not targets or any(type(i) is not int for i in targets + controls):
        raise ValueError("MFI requires nonempty integer target indices and integer control indices")
    if len(set(targets)) != len(targets):
        raise ValueError("MFI target indices must be unique")
    if relative and not controls:
        raise ValueError("MFI relative positioning requires control indices")
    if len(thresholds) != 2 or any(type(v) is not int or v <= 0 for v in thresholds):
        raise ValueError("MFI interpolation thresholds must be two positive integers")
    expanded = targets
    if interpolate or save_interpolated:
        knots = sorted(set(targets + controls))
        generated = set(targets)
        for start, end in zip(knots, knots[1:]):
            threshold = thresholds[0] if controls and start < min(controls) else thresholds[1]
            count = math.ceil((end - start) / threshold)
            generated.update(round(start + (end - start) * j / count) for j in range(1, count))
        generated -= set(controls) - set(targets)
        expanded = tuple(sorted(generated))
    output_indices = expanded if save_interpolated else targets
    origin = min(controls) if relative else 0
    return MFIPlan(
        tuple(i - origin for i in expanded),
        tuple(i - origin for i in controls),
        tuple(expanded.index(i) for i in output_indices),
        output_indices,
        origin,
    )


def cached_indices(batch, name, override):
    value = batch.get(name)
    if value is None:
        return override
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"{name} must be an int64 [1,N] tensor")
    indices = tuple(value[0].tolist())
    if override is not None and tuple(override) != indices:
        raise ValueError(f"{name} cache indices {indices} conflict with CLI indices {tuple(override)}")
    return indices


def mask_condition(latent, mask):
    """A white mask preserves the condition; black zeros its normalized latent."""
    mask = torch.as_tensor(mask, device=latent.device, dtype=latent.dtype)
    if mask.ndim != 2 or not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("MFI control mask must be a finite [H,W] tensor in [0,1]")
    mask = F.interpolate(mask[None, None], size=latent.shape[-2:], mode="area")
    return latent * mask.unsqueeze(2)
