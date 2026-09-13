"""Triton implementations of Neurite tensor operations."""

from __future__ import annotations

import torch

try:
    import triton as tri
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in installations without Triton
    tri = None
    tl = None


if tri is not None:

    @tri.jit
    def _connected_components_initialize_parents(
        mask_pointer,
        parent_pointer,
        numel: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        active = offsets < numel
        foreground = tl.load(mask_pointer + offsets, mask=active, other=0).to(tl.int1)
        parents = tl.where(foreground, offsets, -1)
        tl.store(parent_pointer + offsets, parents, mask=active)

    @tri.jit
    def _connected_components_hook_neighbors(
        parent_pointer,
        changed_pointer,
        depth: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        ndim: tl.constexpr,
        connectivity: tl.constexpr,
        block_size: tl.constexpr,
    ):
        plane_size = height * width
        numel = depth * plane_size
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        active = offsets < numel
        depths = offsets // plane_size
        plane_offsets = offsets % plane_size
        rows = plane_offsets // width
        columns = plane_offsets % width
        parents = tl.load(parent_pointer + offsets, mask=active, other=-1)
        foreground = active & (parents >= 0)
        changed_pointers = changed_pointer + tl.zeros((block_size,), dtype=tl.int32)
        changed_values = tl.full((block_size,), 1, dtype=tl.int32)

        # Visit one direction for every undirected edge. Hooks always point
        # toward smaller IDs, so the parent forest cannot contain a cycle.
        for depth_offset in tl.static_range(-1, 2):
            for row_offset in tl.static_range(-1, 2):
                for column_offset in tl.static_range(-1, 2):
                    distance = (depth_offset != 0) + (row_offset != 0) + (column_offset != 0)
                    earlier = (depth_offset < 0) or (
                        depth_offset == 0
                        and ((row_offset < 0) or (row_offset == 0 and column_offset < 0))
                    )
                    valid_dimension = (ndim >= 3 or depth_offset == 0) and (
                        ndim >= 2 or row_offset == 0
                    )
                    if earlier and valid_dimension and distance <= connectivity:
                        neighbor_depths = depths + depth_offset
                        neighbor_rows = rows + row_offset
                        neighbor_columns = columns + column_offset
                        neighbor_active = (
                            foreground
                            & (neighbor_depths >= 0)
                            & (neighbor_depths < depth)
                            & (neighbor_rows >= 0)
                            & (neighbor_rows < height)
                            & (neighbor_columns >= 0)
                            & (neighbor_columns < width)
                        )
                        neighbor_offsets = (
                            offsets + depth_offset * plane_size + row_offset * width + column_offset
                        )
                        neighbor_parents = tl.load(
                            parent_pointer + neighbor_offsets,
                            mask=neighbor_active,
                            other=-1,
                        )
                        pair = (
                            neighbor_active
                            & (neighbor_parents >= 0)
                            & (neighbor_parents != parents)
                        )
                        high = tl.maximum(parents, neighbor_parents)
                        low = tl.minimum(parents, neighbor_parents)
                        previous = tl.atomic_min(parent_pointer + high, low, mask=pair)
                        pair_changed = pair & (previous > low)
                        tl.atomic_max(changed_pointers, changed_values, mask=pair_changed)

    @tri.jit
    def _connected_components_compress_parents(
        parent_pointer,
        changed_pointer,
        numel: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        active = offsets < numel
        parents = tl.load(parent_pointer + offsets, mask=active, other=-1)
        foreground = active & (parents >= 0)
        grandparents = tl.load(parent_pointer + parents, mask=foreground, other=-1)
        needs_compression = foreground & (grandparents != parents)
        tl.store(parent_pointer + offsets, grandparents, mask=needs_compression)
        changed_pointers = changed_pointer + tl.zeros((block_size,), dtype=tl.int32)
        changed_values = tl.full((block_size,), 1, dtype=tl.int32)
        tl.atomic_max(changed_pointers, changed_values, mask=needs_compression)


def is_available() -> bool:
    """Return whether the optional Triton dependency is available."""
    return tri is not None


def connected_components(
    mask: torch.Tensor,
    connectivity: int = 1,
) -> tuple[torch.Tensor, int]:
    """
    Label connected components in a spatial CUDA mask using Triton.

    Parameters
    ----------
    mask : torch.Tensor
        Spatial binary mask of shape [*V] on a CUDA device.
    connectivity : int, default=1
        Neighborhood connectivity from one through the number of spatial
        dimensions. One uses face connectivity; `mask.ndim` uses full
        connectivity.

    Returns
    -------
    tuple[torch.Tensor, int]
        Integer component labels with shape [*V] and the number of components.
        Background is zero and components are numbered from one.

    Raises
    ------
    RuntimeError
        If Triton is unavailable or labeling does not converge.
    ValueError
        If the mask, connectivity, CUDA device, or tensor size is unsupported.
    """
    max_elements = torch.iinfo(torch.int32).max
    if mask.numel() > max_elements:
        raise ValueError("Triton connected components supports at most int32 elements")
    if tri is None:
        raise RuntimeError("Triton connected components requires Triton")
    if mask.device.type != "cuda":
        raise ValueError("Triton connected components requires a CUDA tensor")
    if mask.ndim not in (1, 2, 3):
        raise ValueError("mask must have one, two, or three spatial dimensions")
    if connectivity < 1 or connectivity > mask.ndim:
        raise ValueError("connectivity must be between 1 and mask.ndim")
    if mask.numel() == 0:
        return torch.zeros_like(mask, dtype=torch.long), 0

    mask = mask.bool().contiguous()
    normalized_shape = (1,) * (3 - mask.ndim) + tuple(mask.shape)
    depth, height, width = normalized_shape
    numel = mask.numel()
    parents = torch.empty(numel, dtype=torch.int32, device=mask.device)  # [N]
    changed = torch.empty((), dtype=torch.int32, device=mask.device)
    block_size = 256
    grid = (tri.cdiv(numel, block_size),)
    _connected_components_initialize_parents[grid](
        mask,
        parents,
        numel=numel,
        block_size=block_size,
    )

    # Only the scalar convergence flag crosses to Python. Spatial tensors stay
    # on the GPU throughout labeling and scan-order remapping.
    max_iterations = 64
    for _ in range(max_iterations):
        changed.zero_()
        _connected_components_hook_neighbors[grid](
            parents,
            changed,
            depth=depth,
            height=height,
            width=width,
            ndim=mask.ndim,
            connectivity=connectivity,
            block_size=block_size,
        )
        _connected_components_compress_parents[grid](
            parents,
            changed,
            numel=numel,
            block_size=block_size,
        )
        if not bool(changed):
            break
    else:
        message = f"connected components did not converge in {max_iterations} iterations"
        raise RuntimeError(message)

    foreground = parents >= 0  # [N]
    roots = parents[foreground]  # [Nfg]
    unique_roots, inverse = torch.unique(roots, sorted=True, return_inverse=True)
    components = torch.zeros(numel, dtype=torch.long, device=mask.device)  # [N]
    components[foreground] = inverse + 1

    return components.reshape(mask.shape), unique_roots.numel()


__all__ = ["connected_components", "is_available"]
