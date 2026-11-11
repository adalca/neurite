"""Shared numerical contracts, cropped sampling, and default-device behavior."""

import math

import pytest
import torch
import torch.nn.functional as F

import neurite as ne
import neurite.nn.functional as nef


@pytest.mark.parametrize("function", [ne.zscore, nef.zscore])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape, dims", [((1,), None), ((2, 1, 3), (1,)), ((), None)])
def test_zscore_singletons_and_gradients(function, unbiased, dtype, shape, dims):
    """Constant singleton groups return zero without invalid backward values."""
    value = torch.arange(math.prod(shape), dtype=dtype)
    value = value.reshape(shape).requires_grad_()

    result = function(value, dim=dims, unbiased=unbiased)
    result.sum().backward()

    assert torch.equal(result, torch.zeros_like(value))
    assert torch.equal(value.grad, torch.zeros_like(value))


@pytest.mark.parametrize("function", [ne.zscore, nef.zscore])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_zscore_variance_choice_preserves_ordinary_values(function, unbiased, dtype):
    """Keep the original sample correction while supporting population variance."""
    value = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
    mean = value.mean((1, 2), keepdim=True)
    std = value.std((1, 2), unbiased=unbiased, keepdim=True)
    expected = (value - mean) / std.clamp_min(1e-8)

    actual = function(value, dim=(1, 2), unbiased=unbiased)

    assert torch.equal(actual, expected)


def test_zscore_singleton_still_validates_reduction_dimensions():
    """The singleton shortcut must not hide an invalid scalar reduction axis."""
    value = torch.ones((), dtype=torch.float16)

    with pytest.raises(IndexError):
        ne.zscore(value, dim=1)


@pytest.mark.parametrize("function, shape", [
    (ne.volshape_to_ndgrid, (3, 4)),
    (nef.volshape_to_ndgrid, (1, 1, 3, 4)),
])
@pytest.mark.parametrize("normalize", [False, True])
def test_coordinate_grid_respects_device_context(function, shape, normalize):
    """Default allocations follow Torch, and explicit devices still override it."""
    with torch.device("meta"):
        default = function(shape, normalize=normalize, stack=True)
        explicit = function(shape, normalize=normalize, stack=True, device="cpu")

    assert default.device.type == "meta"
    assert explicit.device.type == "cpu"
    assert default.shape == explicit.shape == (2, 3, 4)


@pytest.mark.parametrize("ndim", [0, 4, -1])
def test_interpolation_mode_rejects_unsupported_dimensions(ndim):
    """An invalid dimensionality raises an error at the lookup."""
    with pytest.raises(ValueError, match="1, 2, or 3"):
        ne.utils.infer_linear_interpolation_mode(ndim)


@pytest.mark.parametrize("method", ["dense", "separable"])
@pytest.mark.parametrize("normalize", ["sum", "gaussian", None])
@pytest.mark.parametrize("sigma", [0, (0, 0)])
def test_zero_sigma_is_identity(method, normalize, sigma):
    """A zero-width kernel has unit mass and preserves the input and its gradient."""
    value = torch.arange(20, dtype=torch.float64).reshape(1, 1, 4, 5)
    value.requires_grad_(True)
    options = {"sigma": sigma, "method": method, "normalize": normalize}

    result = nef.gaussian_smoothing(value, **options)
    result.sum().backward()

    assert torch.equal(result, value)
    assert torch.equal(value.grad, torch.ones_like(value))


@pytest.mark.parametrize("method", ["dense", "separable"])
@pytest.mark.parametrize("normalize", ["sum", "gaussian", None])
def test_zero_sigma_leaves_only_that_axis_unsmoothed(method, normalize):
    """An anisotropic zero width acts as an impulse, including Gaussian normalization."""
    value = torch.arange(35, dtype=torch.float64).reshape(1, 1, 5, 7)
    kernel = ne.gaussian_kernel(0.8, ndim=1, normalize=normalize, dtype=value.dtype)
    weight = kernel.reshape(1, 1, 1, -1)
    radius = kernel.numel() // 2
    padded = F.pad(value, (radius, radius, 0, 0), mode="replicate")
    expected = F.conv2d(padded, weight)
    options = {"sigma": (0, 0.8), "normalize": normalize, "padding_mode": "replicate"}

    result = nef.gaussian_smoothing(value, method=method, **options)

    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("sigma", [-1, float("nan"), float("inf")])
def test_gaussian_kernel_rejects_invalid_sigma(sigma):
    """Reject invalid widths before building coordinates or allocating a kernel."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ne.gaussian_kernel(sigma, ndim=2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("shape", [(9,), (7, 9), (5, 7, 9)])
def test_upsample_crop_matches_dense_values_and_gradients(device, dtype, shape):
    """Central cropping preserves interpolation values and backward derivatives."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    options = {"device": device, "dtype": dtype, "requires_grad": True}
    coarse = torch.randn(2, 3, *([4] * len(shape)), **options)
    size = tuple(value + 5 for value in shape)
    mode = ne.utils.infer_linear_interpolation_mode(len(shape))
    expanded = F.interpolate(coarse, size=size, mode=mode, align_corners=False)
    slices = tuple(slice(2, value + 2) for value in shape)
    expected = expanded[(slice(None), slice(None), *slices)]
    gradient = torch.ones_like(expected)

    actual = ne.upsample_crop(coarse, size, shape, non_spatial_dims=(0, 1))
    dense_gradient = torch.autograd.grad(expected, coarse, gradient, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, coarse, gradient)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, dense_gradient)


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_upsample_crop_preserves_nonspatial_dimensions(leading):
    """The top-level operator accepts pure spatial fields and leading field dimensions."""
    value = torch.ones((*leading, 4, 5))
    non_spatial_dims = tuple(range(len(leading))) or None

    result = ne.upsample_crop(value, (13, 14), (7, 8), non_spatial_dims)

    assert result.shape == (*leading, 7, 8)
    assert torch.equal(result, torch.ones_like(result))


def test_upsample_crop_bounds_expanded_field_allocations(monkeypatch):
    """Ordinary cropped interpolation expands only one independent field at a time."""
    original = F.interpolate
    calls = []
    value = torch.ones(2, 3, 4, 5)

    def tracked(input_tensor, *args, **kwargs):
        calls.append(input_tensor.shape[:2])

        return original(input_tensor, *args, **kwargs)

    monkeypatch.setattr(F, "interpolate", tracked)

    result = nef.upsample_crop(value, (15, 17), (7, 9))

    assert result.shape == (2, 3, 7, 9)
    assert calls == [(1, 1)] * 6


def test_upsample_crop_direct_sampling_matches_dense(monkeypatch):
    """Check the oversized-grid path against an affordable interpolation reference."""
    monkeypatch.setattr(nef, "_TRILINEAR_MAX_ELEMENTS", 1)
    coarse = torch.randn(2, 3, 4, 6, 8, requires_grad=True)
    expanded = F.interpolate(coarse, size=(23, 27, 31), mode="trilinear", align_corners=False)
    expected = expanded[:, :, 4:19, 5:22, 6:25]
    gradient = torch.randn_like(expected)

    result = nef.upsample_crop(coarse, (23, 27, 31), (15, 17, 19))
    first = torch.autograd.grad(expected, coarse, gradient, retain_graph=True)[0]
    second = torch.autograd.grad(result, coarse, gradient)[0]

    torch.testing.assert_close(result, expected, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(second, first, rtol=3e-4, atol=3e-4)


def test_upsample_crop_exceeds_int32_grid_without_full_allocation():
    """A linear field has known values even when the padded grid exceeds 2**31 voxels."""
    coarse = torch.arange(27).float().reshape(1, 1, 3, 3, 3).requires_grad_()
    shape = (8, 9, 10)
    size = tuple(value + 1480 for value in shape)
    coordinates = []
    for cropped, full in zip(shape, size):
        index = torch.arange(cropped).float()
        coordinates.append((index + 740 + 0.5) * (3 / full) - 0.5)
    depth, height, width = torch.meshgrid(*coordinates, indexing="ij")
    expected = 9 * depth + 3 * height + width

    result = nef.upsample_crop(coarse, size, shape)
    result.sum().backward()

    assert math.prod(size) > 2**31
    torch.testing.assert_close(result[0, 0], expected, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(coarse.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [(2049, 2049, 513), (200001, 201, 201)])
def test_large_grid_low_precision_center_and_gradient(device, dtype, size):
    """Large-grid sampling retains voxel centers beyond low-precision coordinate limits."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    coarse = torch.arange(27, dtype=dtype, device=device).reshape(1, 1, 3, 3, 3)
    coarse.requires_grad_()
    expected_gradient = torch.zeros_like(coarse)
    expected_gradient[:, :, 1, 1, 1] = 1

    result = nef.upsample_crop(coarse, size, (1, 1, 1))
    result.sum().backward()

    assert result.dtype == dtype
    assert result.device == coarse.device
    assert result.item() == 13
    torch.testing.assert_close(coarse.grad, expected_gradient)


@pytest.mark.parametrize("name, options", [
    ("upsample_noise", {"scale": 3}),
    ("fractal_noise", {"scales": [2, 4], "method": "upsample", "standardize": False}),
    ("fractal_noise", {"scales": [0.5, 1], "method": "blur", "standardize": False}),
])
@pytest.mark.parametrize("shape", [(9,), (7, 9), (5, 7, 9)])
def test_cropped_noise_preserves_full_grid_draws(name, options, shape):
    """Cropping changes storage, while retaining full-grid scales and random streams."""
    function = getattr(ne, name)
    size = tuple(value + 6 for value in shape)
    first = torch.Generator().manual_seed(127)
    second = torch.Generator().manual_seed(127)
    arguments = {"shape": (2, 3, *size), "non_spatial_dims": (0, 1), **options}
    full = function(generator=first, **arguments)
    slices = tuple(slice(3, value + 3) for value in shape)
    expected = full[(slice(None), slice(None), *slices)]

    actual = function(crop_shape=shape, generator=second, **arguments)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(first.get_state(), second.get_state())


def test_cropped_fractal_standardizes_the_returned_region():
    """Final noise statistics describe the crop rather than the discarded exterior."""
    generator = torch.Generator().manual_seed(127)
    options = {"shape": (2, 3, 19, 21), "scales": [2, 4], "method": "upsample"}
    options.update(crop_shape=(9, 11), non_spatial_dims=(0, 1), generator=generator)

    result = ne.fractal_noise(**options)

    torch.testing.assert_close(result.mean((-2, -1)), torch.zeros(2, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.std((-2, -1)), torch.ones(2, 3))


@pytest.mark.parametrize("size, crop_shape", [((0, 8), (2, 3)), ((8,), (2, 3)), ((8, 8), (9, 3))])
def test_upsample_crop_rejects_invalid_shapes(size, crop_shape):
    """Reject nonpositive, inconsistent, or oversized crops."""
    value = torch.ones(1, 1, 4, 4)

    with pytest.raises(ValueError):
        nef.upsample_crop(value, size, crop_shape)


@pytest.mark.parametrize("function, shape", [
    (ne.fractal_noise, (9, 11)), (nef.fractal_noise, (1, 1, 9, 11)),
])
@pytest.mark.parametrize("method", ["blur", "upsample"])
@pytest.mark.parametrize("crop_shape, error", [
    (0, ValueError), (-1, ValueError), ((0, 3), ValueError),
    ((10, 3), ValueError), ((3,), ValueError), ((1.5, 3), TypeError), (True, TypeError),
])
def test_fractal_noise_rejects_invalid_crops_before_sampling(
    function, shape, method, crop_shape, error,
):
    """Both methods reject invalid crops without advancing the caller's random stream."""
    generator = torch.Generator().manual_seed(127)
    original_state = generator.get_state().clone()
    options = {"scales": [1, 2], "method": method, "crop_shape": crop_shape}

    with pytest.raises(error):
        function(shape, generator=generator, **options)

    assert torch.equal(generator.get_state(), original_state)
