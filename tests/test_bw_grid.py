"""Grid values, input errors, and device behavior for both public entry points."""

import pytest
import torch

import neurite as ne
import neurite.nn.functional as nef


@pytest.fixture(params=[ne.bw_grid, nef.bw_grid], ids=["neurite", "nn.functional"])
def grid_function(request):
    """Exercise the public wrapper and implementation with the same cases."""
    return request.param


def test_bw_grid_matches_expected_lines(grid_function):
    """Preserve line spacing and the forced final row and column."""
    expected = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.float32)

    grid = grid_function((6, 7), spacing=2, thickness=1)

    assert torch.equal(grid, expected)


def test_bw_grid_thick_lines_fill_volume(grid_function):
    """Cover every voxel when one axis has no gap between thick grid lines."""
    expected = torch.ones(5, 6, 4)

    grid = grid_function((5, 6, 4), spacing=(1, 2, 1), thickness=2)

    assert torch.equal(grid, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.bool])
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_bw_grid_shapes_boundaries_and_options(grid_function, device, dtype, indexing):
    """Preserve scalar/list shapes, anisotropic spacing, and zero-spacing grids."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    options = {"device": device, "dtype": dtype, "indexing": indexing}

    # Scalar shapes use spacing-plus-one periods and include the final pixel.
    actual = grid_function(8, spacing=2, **options)
    expected = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1], device=device, dtype=dtype)
    assert torch.equal(actual, expected)
    assert actual.device == expected.device
    assert actual.dtype == expected.dtype

    # Rectangular grids accept lists and different spacing along each axis.
    full_row = [1] * 8
    sparse_row = [1, 1, 0, 0, 1, 1, 0, 1]
    rows = [full_row, full_row, sparse_row, full_row, full_row]
    expected = torch.tensor(rows, device=device, dtype=dtype)
    actual = grid_function([5, 8], spacing=[2, 3], thickness=2, **options)
    assert torch.equal(actual, expected)

    # The center is the only voxel outside grid lines in this cube.
    expected = torch.ones(3, 3, 3, device=device, dtype=dtype)
    expected[1, 1, 1] = 0
    actual = grid_function((3, 3, 3), spacing=1, **options)
    assert torch.equal(actual, expected)

    # Zero spacing on any axis fills every voxel, including singleton dimensions.
    expected = torch.ones(1, 4, device=device, dtype=dtype)
    actual = grid_function((1, 4), spacing=(3, 0), **options)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("spacing", [0, 2])
def test_bw_grid_accepts_string_dtype(grid_function, spacing):
    """Resolve string dtypes in both ordinary and zero-spacing grids."""
    grid = grid_function((4, 5), spacing=spacing, dtype="float64")

    assert grid.dtype == torch.float64


@pytest.mark.parametrize("options", [{}, {"device": None}], ids=["omitted", "none"])
def test_bw_grid_uses_default_device(grid_function, options):
    """A non-CPU device context controls allocation without requiring a GPU."""
    with torch.device("meta"):
        grid = grid_function((4, 5), spacing=0, **options)

    assert grid.device.type == "meta"
    assert grid.shape == (4, 5)


@pytest.mark.parametrize("spacing", [0, 2])
def test_bw_grid_explicit_device_overrides_default(grid_function, spacing):
    """An explicit CPU device overrides a non-CPU default for all allocations."""
    with torch.device("meta"):
        grid = grid_function((4, 5), spacing=spacing, device="cpu")

    assert grid.device.type == "cpu"
    assert grid.shape == (4, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("spacing", [0, 2])
@pytest.mark.parametrize("options", [{}, {"device": None}], ids=["omitted", "none"])
def test_bw_grid_uses_default_cuda_device(grid_function, spacing, options):
    """Use the selected GPU for both zero-spacing and periodic grids."""
    expected = grid_function((4, 5), spacing=spacing, device="cpu")

    with torch.device("cuda"):
        grid = grid_function((4, 5), spacing=spacing, **options)

    assert grid.device.type == "cuda"
    assert torch.equal(grid.cpu(), expected)


@pytest.mark.parametrize("shape, spacing, thickness, indexing, error, message", [
    ((), 2, 1, "ij", ValueError, "positive dimensions"),
    ((0, 5), 2, 1, "ij", ValueError, "positive dimensions"),
    ((-1, 5), 2, 1, "ij", ValueError, "positive dimensions"),
    ((4.5, 5), 2, 1, "ij", TypeError, "vol_shape.*integers"),
    ((True, 5), 2, 1, "ij", TypeError, "vol_shape.*integers"),
    ((4, 5), (2,), 1, "ij", ValueError, "same length"),
    ((4, 5), (1, 2, 3), 1, "ij", ValueError, "same length"),
    ((4, 5), -1, 1, "ij", ValueError, "nonnegative"),
    ((4, 5), (2, 1.5), 1, "ij", TypeError, "spacing.*integers"),
    ((4, 5), True, 1, "ij", TypeError, "spacing.*integers"),
    ((4, 5), 2, 0, "ij", ValueError, "thickness.*positive"),
    ((4, 5), 0, 1.5, "ij", TypeError, "thickness.*integer"),
    ((4, 5), 2, True, "ij", TypeError, "thickness.*integer"),
    ((4, 5), 2, 1, "invalid", ValueError, "indexing"),
    ((4, 5), 0, 1, "invalid", ValueError, "indexing"),
    ((0, 5), 0, 1, "ij", ValueError, "positive dimensions"),
    ((4, 5), (0, -1), 1, "ij", ValueError, "nonnegative"),
])
def test_bw_grid_rejects_invalid_inputs(
    grid_function, shape, spacing, thickness, indexing, error, message,
):
    """Validate all arguments before either rendering path allocates its output."""
    with pytest.raises(error, match=message):
        grid_function(shape, spacing, thickness=thickness, indexing=indexing)
