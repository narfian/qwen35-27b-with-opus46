"""Runtime monkey-patches applied before training starts.

These patches work around third-party incompatibilities that we cannot fix
upstream from this project. Each patch is idempotent and silently degrades
to a no-op when its target is unavailable (e.g. an optional dependency is
not installed, or the running GPU does not need the workaround).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ``flash-linear-attention`` ships Triton autotune candidates that include
# ``num_stages >= 3`` for Ampere. The kernels below request more shared memory
# than A100/RTX-30xx expose (max ~166,912 B per SM) and crash with
# ``triton.runtime.errors.OutOfResources: shared memory``. We trim those
# candidates so only Ampere-safe configs survive.
_FLA_DELTA_H_KERNELS: tuple[str, ...] = (
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
)


def apply_fla_ampere_safe_autotune(max_num_stages: int = 2) -> bool:
    """Cap Triton ``num_stages`` for FLA chunk_gated_delta_rule kernels.

    Returns True when at least one kernel was patched, False otherwise
    (no CUDA, GPU is Hopper/Blackwell, FLA not installed, kernel layout
    changed in a future FLA release, etc.).
    """

    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        cap_major = torch.cuda.get_device_capability(0)[0]
    except Exception:  # noqa: BLE001 - defensive against driver issues
        return False

    if cap_major >= 9:
        # Hopper(9.x) / Blackwell(10.x) have enough shared memory; leave the
        # full autotune candidate set in place so they can pick the fastest.
        return False

    try:
        from fla.ops.common import chunk_delta_h
    except ImportError:
        logger.debug("flash-linear-attention not installed; skipping FLA patch")
        return False

    patched: list[tuple[str, int, int]] = []
    for kernel_name in _FLA_DELTA_H_KERNELS:
        kernel = getattr(chunk_delta_h, kernel_name, None)
        if kernel is None:
            continue
        autotuner = _walk_to_autotuner(kernel)
        if autotuner is None:
            continue
        before = len(autotuner.configs)
        new_configs = [
            cfg
            for cfg in autotuner.configs
            if getattr(cfg, "num_stages", 0) <= max_num_stages
        ]
        if not new_configs or len(new_configs) == before:
            continue
        autotuner.configs = new_configs
        cache = getattr(autotuner, "cache", None)
        if isinstance(cache, dict):
            cache.clear()
        patched.append((kernel_name, before, len(new_configs)))

    if patched:
        details = ", ".join(f"{name}: {b}->{a}" for name, b, a in patched)
        print(
            f"[runtime_patch] FLA autotune capped to num_stages<={max_num_stages} "
            f"for Ampere-safe shared memory ({details})"
        )
    else:
        logger.debug(
            "FLA Ampere-safe autotune patch: no kernels needed trimming"
        )
    return bool(patched)


def _walk_to_autotuner(kernel: Any) -> Any | None:
    """Walk inward through ``Heuristics(Autotuner(JITFunction))`` wrappers.

    Returns the first object exposing a list-typed ``configs`` attribute
    (i.e. the underlying ``triton.runtime.autotuner.Autotuner``).
    """

    node: Any = kernel
    for _ in range(8):
        if node is None:
            return None
        configs = getattr(node, "configs", None)
        if isinstance(configs, list):
            return node
        node = getattr(node, "fn", None)
    return None
