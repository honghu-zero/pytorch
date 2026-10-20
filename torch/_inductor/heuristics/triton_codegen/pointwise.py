"""Codegen heuristics for Triton pointwise kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the dimension methods that
differ from the base.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import TileHint, TRITON_MAX_BLOCK


if TYPE_CHECKING:
    from collections.abc import Callable


def _pointer_element_bytes(arg_type: object) -> int:
    """Bytes per element of a pointer argument in a Triton signature (2 for "*bf16",
    4 for a pointer type that names no width), and 0 for any other argument."""
    if not isinstance(arg_type, str) or not arg_type.startswith("*"):
        return 0
    bits = re.search(r"\d+", arg_type)
    return max(1, int(bits.group()) // 8) if bits else 4


# ----------------------------------------------------------------------
# Default (CUDA) pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise")
class PointwiseHeuristic(CodegenConfigHeuristics):
    """Default pointwise autotuning configs (CUDA non-ROCm)."""

    # (XBLOCK, num_warps) candidates of a kernel on a capped grid, with the warp counts
    # for 32-lane GPUs (halved on 64-lane ones, for the same threads per program). Each
    # program loops over many x blocks, and a large block keeps enough loads in flight.
    # The values come from one H100 measurement, a bf16-to-fp16 cast at 44 and 66
    # programs. The table stops at 32,768 elements because 65,536-element blocks
    # spilled registers on an A100 in every kernel tried. A device heuristic can
    # override them.
    capped_grid_candidates: tuple[tuple[int, int], ...] = (
        (16384, 16),
        (16384, 32),
        (32768, 8),
        (32768, 16),
    )
    # The most bytes of the kernel's tensors that one candidate block may hold: half of
    # an NVIDIA SM's 65,536-register file, which leaves the other half for the kernel's
    # other values. The bound is per block, not per thread, because a candidate with
    # more warps gives each thread fewer registers.
    capped_grid_bytes_per_block: int = 131072

    def get_configs(
        self,
        size_hints: dict[str, int],
        bs: int,
        triton_config_fn: Callable[..., Any],
        hinted_configs: list[Any],
        tile_hint: TileHint | None = None,
        inductor_meta: dict[str, Any] | None = None,
    ) -> list[Any]:
        inductor_meta = inductor_meta or {}
        n = len(size_hints)
        if n == 1:
            return self._configs_1d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        if n == 2:
            return self._configs_2d(
                size_hints,
                bs,
                hinted_configs,
                triton_config_fn,
                tile_hint,
                inductor_meta,
            )
        if n == 3:
            return self._configs_3d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        raise NotImplementedError(f"size_hints: {size_hints}")

    def get_capped_grid_configs(
        self,
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
        min_elem_per_thread: int = 0,
    ) -> list[Any]:
        """Configs for a pointwise kernel on a capped grid, which replace get_configs'.

        The default keeps the `capped_grid_candidates` whose XBLOCK is at most
        `grid_max_xblock`, leaves each of the `grid_cap` programs at least one block of
        the size hint (for a dynamic size, its compile-time example) and holds at most
        `capped_grid_bytes_per_block` bytes of the kernel's tensors; if none does, one
        config at the largest such XBLOCK. As in triton_config, XBLOCK then grows to
        give every thread at least `min_elem_per_thread` elements, past
        `grid_max_xblock` if need be, up to the size hint and the kernel's max block
        (see check_capped_grid_config). Where XBLOCK cannot grow, the warps shrink,
        down to one; with 32-lane warps outside ROCm a block of 128 elements or more
        keeps at least four, and XBLOCK grows again when that floor adds warps.
        """
        from torch._inductor.runtime.runtime_utils import last_power_of_2
        from torch._inductor.runtime.triton_heuristics import (
            capped_grid_max_block,
            capped_grid_max_xblock,
            validate_grid_cap,
        )

        # triton_compat's Config is a stand-in when Triton is missing, so type checkers
        # read Triton's own
        if TYPE_CHECKING:
            from triton import Config
        else:
            from torch._inductor.runtime.triton_compat import Config

        device = triton_meta["device"]
        warp_size = device.warp_size_or_default
        max_warps = last_power_of_2(
            max(1, (device.max_threads_per_block or 1024) // warp_size)
        )
        # Halve the warps on 64-lane GPUs (down to one) for the same threads per program
        warp_scale = max(1, warp_size // 32)
        floor_warps = warp_size == 32 and not torch.version.hip

        def warps(xblock: int, num_warps: int) -> int:
            # At least 4 warps for 128 elements or more with 32-lane warps outside ROCm,
            # as triton_config gives them to avoid a ptxas bug
            if floor_warps and xblock >= 128:
                num_warps = max(num_warps, 4)
            return max(1, min(num_warps, max_warps))

        grid_cap = validate_grid_cap(inductor_meta.get("grid_cap"))
        max_xblock = min(
            capped_grid_max_xblock(inductor_meta),
            last_power_of_2(max(1, size_hints["x"] // grid_cap)),
        )
        # Bytes per element of the kernel's tensors: its pointer arguments' element
        # sizes, summed
        element_bytes = sum(
            _pointer_element_bytes(arg_type)
            for arg_type in triton_meta.get("signature", {}).values()
        )
        if element_bytes:
            max_xblock = min(
                max_xblock,
                last_power_of_2(
                    max(1, self.capped_grid_bytes_per_block // element_bytes)
                ),
            )
        candidates = [
            (xblock, max(1, min(num_warps // warp_scale, max_warps)))
            for xblock, num_warps in self.capped_grid_candidates
            if xblock <= max_xblock
        ]
        if not candidates:
            # One warp per 128 elements, as triton_config_reduction gives, halved on
            # 64-lane GPUs like the candidates' warps
            candidates = [
                (max_xblock, warps(max_xblock, max_xblock // 128 // warp_scale))
            ]
        # As in _configs_1d: one config without pointwise autotuning or max-autotune
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            candidates = candidates[:1]
        max_block = capped_grid_max_block(inductor_meta)

        def raised(xblock: int, num_warps: int) -> int:
            return min(
                max(xblock, min_elem_per_thread * warp_size * num_warps),
                size_hints["x"],
                max_block,
            )

        pairs: list[tuple[int, int]] = []
        for xblock, num_warps in candidates:
            if min_elem_per_thread:
                # XBLOCK grows for min_elem_per_thread, as in triton_config; where it
                # cannot, the warps shrink, and a floor that adds warps grows it again
                xblock = raised(xblock, num_warps)
                num_warps = warps(
                    xblock, min(num_warps, xblock // (min_elem_per_thread * warp_size))
                )
                xblock = raised(xblock, num_warps)
            # Capped warps and raised blocks can make two candidates equal
            if (xblock, num_warps) not in pairs:
                pairs.append((xblock, num_warps))
        return [
            Config({"XBLOCK": xblock}, num_warps=num_warps, num_stages=1)
            for xblock, num_warps in pairs
        ]

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, bs)]
        return [
            triton_config_fn(size_hints, bs, num_elements_per_warp=256),
            triton_config_fn(size_hints, bs // 2, num_elements_per_warp=64),
            *hinted_configs,
        ]

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        if (
            not inductor_meta.get("autotune_pointwise", True)
            or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        if not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 16, 16, 16)]
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]


# ----------------------------------------------------------------------
# ROCm/HIP pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
class ROCmPointwiseHeuristic(PointwiseHeuristic):
    """Pointwise configs for ROCm/HIP devices.

    Inherits _configs_3d from PointwiseHeuristic (same configs).
    """

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.extend(
                [
                    triton_config_fn(size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2),
                    triton_config_fn(
                        size_hints,
                        4096,  # wrt: better than the max_block for some kernel
                    ),
                    triton_config_fn(
                        size_hints,
                        2048,
                        num_warps=8,
                        num_stages=2,
                        waves_per_eu=1,  # 20% improvement
                    ),
                ]
            )
            if inductor_meta.get("atomic_add_found"):
                configs.append(
                    triton_config_fn(
                        size_hints,
                        64,
                        num_warps=1,
                        num_stages=1,  # 250% improvement
                    )
                )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # ROCm doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            triton_config_fn(size_hints, 64, 32),  # better for some kernels
            triton_config_fn(size_hints, 128, 16),  # +10% for some kernels
            triton_config_fn(size_hints, 128, 32),  # additional 10% more
            triton_config_fn(size_hints, 32, 512),  # +30% for some kernels
        ]


# ----------------------------------------------------------------------
# XPU pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "xpu", register=torch.xpu._is_compiled())
class XPUPointwiseHeuristic(PointwiseHeuristic):
    """Pointwise configs for XPU devices."""

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.append(
                # intel-xpu-backend-for-triton #5124
                triton_config_fn(size_hints, 32)
            )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # XPU doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            # intel-xpu-backend-for-triton #5198
            triton_config_fn(size_hints, 32, 32, num_warps=8),
            # intel-xpu-backend-for-triton #5199
            triton_config_fn(size_hints, 4, 256),
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # XPU always uses full set of configs (no autotune gate)
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]
