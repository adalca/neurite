"""
Tests for vectorized functions in neurite.nn.functional.

These tests verify that vectorized implementations produce identical outputs
to the original loop-based implementations.
"""

import pytest
import torch
from pystrum.pynd.ndutils import bw_grid as pystrum_bw_grid

import neurite as ne
import neurite.nn.functional as nef


# =============================================================================
# Tests for subsample_random_dims()
# =============================================================================

def test_subsample_random_dims_all_dims():
    """Test subsampling when all dimensions are selected."""
    input_tensor = torch.arange(1, 65).reshape(1, 1, 8, 8).float()

    # Force all dimensions to be subsampled (p=1.0)
    # With forbidden_dims=(0, 1), only spatial dims 2 and 3 can be subsampled
    # Both will be subsampled with stride=2, resulting in 4x4 output
    torch.manual_seed(42)
    result = nef.subsample_random_dims(input_tensor, stride=2, p=1.0)

    # Should be downsampled by 2 in both spatial dims
    assert result.shape == (1, 1, 4, 4)

    # Expected output: subsampling stride=2 along both spatial dimensions
    # Input 8x8 grid (values 1-64) -> takes rows/cols at indices 0,2,4,6
    # This results in the 4x4 grid with these exact values:
    expected = torch.tensor([[[
        [1., 3., 5., 7.],
        [17., 19., 21., 23.],
        [33., 35., 37., 39.],
        [49., 51., 53., 55.]
    ]]]).float()

    # Verify exact values match
    assert torch.equal(result, expected), (
        f"Subsampling result does not match expected output.\n"
        f"Input shape: {input_tensor.shape}\n"
        f"Result shape: {result.shape}\n"
        f"Result:\n{result[0, 0]}\n"
        f"Expected:\n{expected[0, 0]}"
    )


def test_subsample_random_dims_no_dims():
    """Test subsampling when no dimensions are selected."""
    input_tensor = torch.randn(1, 1, 8, 8)

    # Force no dimensions to be subsampled (p=0.0)
    result = nef.subsample_random_dims(input_tensor, stride=2, p=0.0)

    # Should remain unchanged
    assert result.shape == input_tensor.shape
    assert torch.equal(result, input_tensor)


def test_subsample_random_dims_3d():
    """Test 3D subsampling."""
    input_tensor = torch.randn(1, 2, 16, 16, 16)

    # Subsample with p=1.0 (all dims)
    result = nef.subsample_random_dims(input_tensor, stride=2, p=1.0)

    # All spatial dims should be halved
    assert result.shape == (1, 2, 8, 8, 8)


def test_subsample_random_dims_deterministic():
    """Test that same seed produces same result."""
    input_tensor = torch.randn(1, 1, 16, 16)

    torch.manual_seed(42)
    result1 = nef.subsample_random_dims(input_tensor, stride=2, p=0.5)

    torch.manual_seed(42)
    result2 = nef.subsample_random_dims(input_tensor, stride=2, p=0.5)

    assert torch.equal(result1, result2)


# =============================================================================
# Tests for sample_image_from_labels()
# =============================================================================

def test_sample_image_from_labels_shape():
    """Test that output shape matches input shape."""
    label_tensor = torch.tensor([
        [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [3, 3, 4, 4],
         [3, 3, 4, 4]]
    ]).unsqueeze(0)

    result = nef.sample_image_from_labels(label_tensor)

    assert result.shape == label_tensor.shape


def test_sample_image_from_labels_constant_within_region():
    """Test that each label region has relatively consistent intensity."""
    label_tensor = torch.tensor([
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [2, 2, 2, 2]]
    ]).unsqueeze(0)

    # Use small noise std for this test
    result = ne.sample_image_from_labels(
        label_tensor,
        mean_range=(0.49, 0.51),  # Fixed mean of 0.5
        noise_std=0.1  # sqrt(0.01) = 0.1
    )

    # Within each region, values should be close (low variance)
    region1_values = result[label_tensor == 1]
    region2_values = result[label_tensor == 2]

    # Standard deviation within region should be small (sqrt(0.01) ≈ 0.1)
    assert region1_values.std() < 0.3  # Allow some margin
    assert region2_values.std() < 0.3


def test_sample_image_from_labels_different_regions():
    """Test that different label regions can have different intensities."""
    label_tensor = torch.tensor([
        [[1, 1, 2, 2],
         [1, 1, 2, 2]]
    ]).unsqueeze(0)

    # Use very small noise to make regions nearly uniform
    torch.manual_seed(42)
    result = nef.sample_image_from_labels(
        label_tensor,
        mean_range=(0, 1),  # Uniform sampling from [0, 1]
        noise_std=0.03  # sqrt(0.001) ≈ 0.03
    )

    # Each region should have low variance
    region1_std = result[label_tensor == 1].std()
    region2_std = result[label_tensor == 2].std()

    # Standard deviation should be very small (sqrt(0.001) ≈ 0.03)
    assert region1_std < 0.1
    assert region2_std < 0.1


def test_sample_image_from_labels_deterministic():
    """Test that same seed produces same result."""
    label_tensor = torch.randint(0, 5, (1, 1, 8, 8))

    torch.manual_seed(42)
    result1 = nef.sample_image_from_labels(
        label_tensor,
        mean_range=(0, 1),  # Uniform sampling from [0, 1]
        noise_std=0.32  # sqrt(0.1) ≈ 0.32
    )

    torch.manual_seed(42)
    result2 = nef.sample_image_from_labels(
        label_tensor,
        mean_range=(0, 1),  # Uniform sampling from [0, 1]
        noise_std=0.32  # sqrt(0.1) ≈ 0.32
    )

    assert torch.allclose(result1, result2, atol=1e-6)


def test_volshape_to_ndgrid_sizes():
    size = (19, 32)
    the_grid = ne.volshape_to_ndgrid(size=size, stack=True)
    # With channels-first convention: (ndim, *spatial)
    assert tuple(the_grid.shape) == (2, 19, 32)


def test_functional_volshape_to_ndgrid_sizes():
    B, C = 2, 3
    size = (B, C, 43, 9, 10)
    the_grid = nef.volshape_to_ndgrid(size=size, stack=True)
    # nef wrapper handles B, C dims internally, still returns (ndim, *spatial)
    assert tuple(the_grid.shape) == (3, 43, 9, 10)


@pytest.mark.parametrize("vol_shape,spacing,thickness", [
    ((6, 7), 2, 1),
    ((5, 6, 4), (1, 2, 1), 2),
])
def test_bw_grid_matches_pystrum(vol_shape, spacing, thickness):
    """Test that bw_grid preserves the legacy pystrum grid convention."""
    grid = ne.bw_grid(vol_shape=vol_shape, spacing=spacing, thickness=thickness)
    expected = torch.as_tensor(pystrum_bw_grid(vol_shape, spacing, thickness), dtype=grid.dtype)

    assert torch.equal(grid, expected)


def test_bw_grid_rejects_spacing_length_mismatch():
    """Test that per-dimension spacing must match the output shape length."""
    with pytest.raises(AssertionError):
        ne.bw_grid((5, 5), spacing=(1, 2, 3))


def test_gaussian_kernel_sums_to_one():
    """Make sure base kernel is normalized (sums to 1)."""
    kernel = ne.gaussian_kernel(sigma=2.5, ndim=2,)
    total = kernel.sum()
    assert torch.allclose(total, torch.tensor(1.0), atol=1e-6)


def test_gaussian_kernel_orthogonal_slice_symmetry():
    """
    Test that orthogonal slices through the center of a Gaussian kernel are identical,
    demonstrating rotational symmetry.
    """
    sigma = 1.0
    truncate = 2
    kernel = ne.gaussian_kernel(sigma=sigma, truncate=truncate, ndim=3)

    center_idx = 2  # Middle of 5x5x5 kernel

    # Get center slices along each axis
    slice_xy = kernel[center_idx, :, :]  # z=center, varying x,y
    slice_xz = kernel[:, center_idx, :]  # y=center, varying x,z
    slice_yz = kernel[:, :, center_idx]  # x=center, varying y,z

    # All center slices should be identical for isotropic Gaussian
    assert torch.allclose(slice_xy, slice_xz, atol=1e-6)
    assert torch.allclose(slice_xy, slice_yz, atol=1e-6)
    assert torch.allclose(slice_xz, slice_yz, atol=1e-6)


def test_gaussian_kernel_center_is_maximum():
    """Test that the center of the Gaussian kernel has the maximum value."""
    # sigma=1.0, truncate=3: kernel_size = 2*int(3*1+0.5)+1 = 7
    kernel_3d = ne.gaussian_kernel(sigma=1.0, ndim=3)
    center_3d = (3, 3, 3)
    assert kernel_3d[center_3d] == kernel_3d.max()

    # sigma=1.5, truncate=3: kernel_size = 2*int(3*1.5+0.5)+1 = 11
    kernel_2d = ne.gaussian_kernel(sigma=1.5, ndim=2)
    center_2d = (5, 5)
    assert kernel_2d[center_2d] == kernel_2d.max()


def test_gaussian_kernel_normalize_none():
    """Test that normalize=None returns unnormalized kernel (does not sum to 1)."""
    kernel = ne.gaussian_kernel(sigma=1.0, ndim=2, normalize=None)

    # Center value should be exp(0) = 1.0 (unnormalized Gaussian peak)
    center = kernel.shape[0] // 2

    assert torch.allclose(kernel[center, center], torch.tensor(1.0), atol=1e-6)

    # Sum should NOT be 1.0 (it will be > 1 for any kernel larger than 1x1)
    assert kernel.sum() > 1.0


def test_gaussian_kernel_normalize_sum_is_default():
    """Test that normalize='sum' produces identical output to the default."""
    kernel_default = ne.gaussian_kernel(sigma=2.0, ndim=3)
    kernel_sum = ne.gaussian_kernel(sigma=2.0, ndim=3, normalize="sum")

    assert torch.allclose(kernel_default, kernel_sum)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gaussian_kernel_normalize_gaussian_center_value(ndim: int):
    """Test that center of gaussian-normalized kernel equals the true PDF peak."""
    sigma = 1.5
    kernel = ne.gaussian_kernel(sigma=sigma, ndim=ndim, normalize="gaussian")

    center = tuple(s // 2 for s in kernel.shape)
    expected_peak = 1.0 / ((2 * torch.pi) ** (ndim / 2) * sigma ** ndim)

    assert torch.allclose(kernel[center], torch.tensor(expected_peak), atol=1e-6)


def test_gaussian_kernel_normalize_gaussian_anisotropic():
    """Test gaussian normalization with per-dimension sigmas."""
    sigma = [0.5, 1.0, 2.0]
    kernel = ne.gaussian_kernel(sigma=sigma, normalize="gaussian")

    center = tuple(s // 2 for s in kernel.shape)
    expected_peak = 1.0 / ((2 * torch.pi) ** 1.5 * 0.5 * 1.0 * 2.0)

    assert torch.allclose(kernel[center], torch.tensor(expected_peak), atol=1e-6)


@pytest.mark.parametrize("shape,non_spatial_dims", [
    ((64, 64), None),
    ((3, 64, 64), (0,)),
    ((2, 3, 32, 32, 32), (0, 1)),
])
def test_random_smoothed_noise_shape_preservation(shape, non_spatial_dims):
    """Test that output shape matches requested shape."""
    torch.manual_seed(42)
    noise = ne.random_smoothed_noise(shape=shape, sigma=2.0, non_spatial_dims=non_spatial_dims)
    assert noise.shape == shape


@pytest.mark.parametrize("magnitude", [1.0, 2.5, 0.1])
def test_random_smoothed_noise_normalization(magnitude):
    """Test that output has zero mean and std equal to magnitude."""
    torch.manual_seed(42)
    noise = ne.random_smoothed_noise(shape=(128, 128), sigma=3.0, magnitude=magnitude)
    assert abs(noise.mean().item()) < 1e-6
    assert abs(noise.std().item() - magnitude) < 1e-6


@pytest.mark.parametrize("sigma_small,sigma_large", [(1.0, 5.0), (0.5, 3.0)])
def test_random_smoothed_noise_smoothness_increases_with_sigma(sigma_small, sigma_large):
    """Test that larger sigma produces smoother noise (lower gradient magnitude)."""
    torch.manual_seed(42)

    noise_small = ne.random_smoothed_noise(shape=(64, 64), sigma=sigma_small)
    noise_large = ne.random_smoothed_noise(shape=(64, 64), sigma=sigma_large)

    def gradient_magnitude(x):
        dx = torch.diff(x, dim=0)
        dy = torch.diff(x, dim=1)
        return (dx ** 2).mean() + (dy ** 2).mean()

    grad_small = gradient_magnitude(noise_small)
    grad_large = gradient_magnitude(noise_large)

    assert grad_large < grad_small


@pytest.mark.parametrize("shape,non_spatial_dims", [
    ((64, 64), None),
    ((3, 64, 64), (0,)),
    ((2, 3, 32, 32, 32), (0, 1)),
])
def test_upsample_noise_shape_preservation(shape, non_spatial_dims):
    """Test that output shape matches requested shape."""
    torch.manual_seed(42)
    noise = ne.upsample_noise(shape=shape, scale=4.0, non_spatial_dims=non_spatial_dims)
    assert noise.shape == shape


@pytest.mark.parametrize("scale_small,scale_large", [(2.0, 16.0), (4.0, 32.0)])
def test_upsample_noise_smoothness_increases_with_scale(scale_small, scale_large):
    """Test that larger scale produces smoother noise (lower gradient magnitude)."""
    torch.manual_seed(42)

    noise_small = ne.upsample_noise(shape=(64, 64), scale=scale_small)
    noise_large = ne.upsample_noise(shape=(64, 64), scale=scale_large)

    def gradient_magnitude(x):
        dx = torch.diff(x, dim=0)
        dy = torch.diff(x, dim=1)
        return (dx ** 2).mean() + (dy ** 2).mean()

    grad_small = gradient_magnitude(noise_small)
    grad_large = gradient_magnitude(noise_large)

    assert grad_large < grad_small


@pytest.mark.parametrize("shape,non_spatial_dims", [
    ((64, 64), None),
    ((3, 64, 64), (0,)),
    ((2, 3, 32, 32, 32), (0, 1)),
])
def test_fractal_noise_shape_preservation(shape, non_spatial_dims):
    """Test that output shape matches requested shape."""
    torch.manual_seed(42)
    noise = ne.fractal_noise(shape=shape, scales=[2.0, 4.0], non_spatial_dims=non_spatial_dims)
    assert noise.shape == shape


@pytest.mark.parametrize("magnitude", [1.0, 2.5, 0.1])
def test_fractal_noise_normalization(magnitude):
    """Test that output has zero mean and std equal to magnitude."""
    torch.manual_seed(42)
    noise = ne.fractal_noise(shape=(128, 128), scales=[2.0, 4.0], magnitude=magnitude)
    assert abs(noise.mean().item()) < 1e-6
    assert abs(noise.std().item() - magnitude) < 1e-6


@pytest.mark.parametrize("reduction", ['mean', 'sum', 'amax', 'amin', 'std', 'var'])
def test_reduce_matches_torch(reduction):
    """Test that reduce() produces identical results to direct torch calls."""
    torch.manual_seed(42)
    tensor = torch.randn(4, 8, 16)

    result = ne.reduce(tensor, reduction=reduction)
    expected = getattr(torch, reduction)(tensor)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("dim,keepdims,expected_shape", [
    (None, False, ()),
    (None, True, (1, 1, 1)),
    (0, False, (8, 16)),
    (0, True, (1, 8, 16)),
    (1, False, (4, 16)),
    ((0, 2), False, (8,)),
    ((0, 2), True, (1, 8, 1)),
])
def test_reduce_output_shape(dim, keepdims, expected_shape):
    """Test that reduce() produces correct output shapes."""
    tensor = torch.randn(4, 8, 16)
    result = ne.reduce(tensor, reduction='mean', dim=dim, keepdims=keepdims)
    assert result.shape == torch.Size(expected_shape)


def test_reduce_none_returns_unchanged():
    """Test that reduction=None returns the tensor unchanged."""
    tensor = torch.randn(4, 8, 16)
    result = ne.reduce(tensor, reduction=None)
    assert result is tensor


@pytest.mark.parametrize("scale_factor,expected_spatial", [
    (2.0, (128, 128)),
    (0.5, (32, 32)),
    (1.0, (64, 64)),
])
def test_resample_scale_factor(scale_factor, expected_spatial):
    """Test that scale_factor correctly scales spatial dimensions."""
    tensor = torch.randn(64, 64)
    result = ne.resample(tensor, scale_factor=scale_factor)
    assert result.shape == torch.Size(expected_spatial)


@pytest.mark.parametrize("target_size", [(32, 32), (128, 128), (50, 100)])
def test_resample_target_size(target_size):
    """Test that size parameter produces exact target shape."""
    tensor = torch.randn(64, 64)
    result = ne.resample(tensor, size=target_size)
    assert result.shape == torch.Size(target_size)


@pytest.mark.parametrize("shape,non_spatial_dims,expected_shape", [
    ((64, 64), None, (128, 128)),
    ((3, 64, 64), (0,), (3, 128, 128)),
    ((2, 3, 64, 64), (0, 1), (2, 3, 128, 128)),
    ((32, 32, 32), None, (64, 64, 64)),
])
def test_resample_preserves_non_spatial_dims(shape, non_spatial_dims, expected_shape):
    """Test that non-spatial dimensions are preserved during resampling."""
    tensor = torch.randn(*shape)
    result = ne.resample(tensor, scale_factor=2.0, non_spatial_dims=non_spatial_dims)
    assert result.shape == torch.Size(expected_shape)


def test_pad_to_multiple_of_preserves_center_values():
    """Test that pad_to_multiple_of center-pads spatial dimensions."""
    tensor = torch.ones(1, 1, 3, 5)
    result = nef.pad_to_multiple_of(tensor, multiple=4)

    assert result.shape == (1, 1, 4, 8)
    assert torch.equal(result[..., 0:3, 1:6], tensor)
    assert result.sum() == tensor.sum()


def test_top_level_pad_to_multiple_of_preserves_non_spatial_dims():
    """Test top-level pad_to_multiple_of with explicit non-spatial dims."""
    tensor = torch.ones(2, 3, 5, 7)
    result = ne.pad_to_multiple_of(tensor, multiple=4, non_spatial_dims=(0, 1))

    assert result.shape == (2, 3, 8, 8)


def test_mask_border_inner_and_outer_2d():
    """Test inner and outer mask borders for a simple square."""
    mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    mask[..., 1:4, 1:4] = True

    inner = nef.mask_border(mask, thickness=1, border_mode="inner")
    outer = nef.mask_border(mask, thickness=1, border_mode="outer")

    expected_inner = torch.zeros_like(mask)
    expected_inner[..., 1:4, 1:4] = True
    expected_inner[..., 2, 2] = False

    expected_outer = torch.ones_like(mask)
    expected_outer[..., 1:4, 1:4] = False

    assert torch.equal(inner, expected_inner)
    assert torch.equal(outer, expected_outer)


def test_top_level_mask_border_handles_pure_spatial_masks():
    """Test top-level mask_border without batch or channel dimensions."""
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[1:4, 1:4] = True
    border = ne.mask_border(mask, thickness=1)

    assert border.shape == mask.shape
    assert border.sum() == 8


def test_normalize_reference_intensity_batches_channels_and_mask_broadcasting():
    """Normalize each batch and channel independently with one-channel masks."""

    image = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 4.0]], [[2.0, 4.0, 6.0, 8.0]]],
            [[[3.0, 6.0, 9.0, 12.0]], [[4.0, 8.0, 12.0, 16.0]]],
        ]
    )
    mask = torch.tensor([[[[True, True, False, False]]]]).expand(2, -1, -1, -1)
    target = torch.tensor([[5.0], [10.0]])

    normalized, scale = nef.normalize_reference_intensity(image, mask, target)
    expanded_mask = mask.expand_as(image)
    references = []
    for batch_index in range(image.shape[0]):
        channel_references = []
        for channel_index in range(image.shape[1]):
            values = normalized[batch_index, channel_index][
                expanded_mask[batch_index, channel_index]
            ]
            channel_references.append(values.mean())
        references.append(torch.stack(channel_references))

    assert normalized.shape == image.shape
    assert scale.shape == (2, 2, 1, 1)
    assert torch.allclose(torch.stack(references), target.expand(2, 2))


def test_normalize_reference_intensity_trims_outliers_and_preserves_gradients():
    """Use a symmetric trimmed mean while retaining autograd support."""

    values = torch.tensor([1.0, 2.0, 3.0, 100.0], requires_grad=True)
    image = values.reshape(1, 1, 2, 2)
    mask = torch.ones_like(image, dtype=torch.bool)

    normalized, scale = nef.normalize_reference_intensity(
        image, mask, target=5.0, trim_fraction=0.25
    )
    normalized.sum().backward()

    assert torch.allclose(scale.flatten(), torch.tensor([2.0]))
    assert image.grad_fn is not None
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(
    ("image", "mask", "kwargs", "error_type"),
    [
        (torch.ones(1, 1, 2, 2, dtype=torch.int16), None, {}, TypeError),
        (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), {}, TypeError),
        (torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2, dtype=torch.bool), {}, ValueError),
        (torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool), {}, ValueError),
        (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool),
         {"trim_fraction": 0.5}, ValueError),
        (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool),
         {"target": 0}, ValueError),
    ],
)
def test_normalize_reference_intensity_rejects_invalid_inputs(
    image, mask, kwargs, error_type
):
    """Reject invalid masks, reference values, targets, and trimming."""

    if mask is None:
        mask = torch.ones_like(image, dtype=torch.bool)
    with pytest.raises(error_type):
        nef.normalize_reference_intensity(image, mask, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_normalize_reference_intensity_cpu_cuda_agreement():
    """Match normalized values and scales between CPU and CUDA."""

    image = torch.rand(2, 2, 8, 9, 10)
    mask = torch.rand(2, 1, 8, 9, 10) > 0.25
    cpu_image, cpu_scale = nef.normalize_reference_intensity(
        image, mask, target=0.859375, trim_fraction=0.1
    )
    cuda_image, cuda_scale = nef.normalize_reference_intensity(
        image.cuda(), mask.cuda(), target=0.859375, trim_fraction=0.1
    )

    assert torch.allclose(cpu_image, cuda_image.cpu(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(cpu_scale, cuda_scale.cpu(), atol=1e-6, rtol=1e-5)


def test_sample_locations_in_mask_samples_each_label():
    """Test that sampled locations carry the correct label ids."""
    torch.manual_seed(0)
    mask = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
    mask[0, 0, 0, 0] = 1
    mask[0, 0, 0, 1] = 1
    mask[0, 1, 2, 1] = 1
    mask[0, 1, 2, 2] = 1

    locs = nef.sample_locations_in_mask(mask, nb_samples=[1, 2], replacement=False)

    assert locs.shape == (1, 3, 3)
    assert locs.dtype == torch.long
    assert locs[0, :, -1].tolist() == [0, 1, 1]
    for loc in locs[0]:
        row, col, label = loc.tolist()
        assert mask[0, label, row, col] > 0


def test_sample_locations_on_border_samples_border_voxels():
    """Test that border-location sampling uses the computed border mask."""
    torch.manual_seed(0)
    mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    mask[..., 1:4, 1:4] = True

    locs = nef.sample_locations_on_border(mask, thickness=1, nb_samples=4, replacement=False)
    border = nef.mask_border(mask, thickness=1)

    assert locs.shape == (1, 4, 3)
    for loc in locs[0]:
        row, col, label = loc.tolist()
        assert label == 0
        assert border[0, 0, row, col]


def test_locs_to_mask_preserves_trailing_empty_labels():
    """Test locs_to_mask can preserve absent trailing label channels."""
    locs = torch.tensor([[[0, 1, 0], [2, 2, 1]]])
    mask = nef.locs_to_mask(locs, vol_shape=(3, 3), nb_labels=3)

    assert mask.shape == (1, 3, 3, 3)
    assert mask[0, 0, 0, 1]
    assert mask[0, 1, 2, 2]
    assert not mask[0, 2].any()


def test_extract_features_at_locs_returns_channelwise_samples():
    """Test feature extraction at integer spatial locations."""
    features = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    locs = torch.tensor([[[0, 0], [2, 3]]])

    result = nef.extract_features_at_locs(features, locs)
    expected = torch.tensor([[[0, 11], [12, 23]]])

    assert torch.equal(result, expected)


def test_sample_features_at_mask_locs_returns_labels():
    """Test feature sampling from mask locations with label ids."""
    torch.manual_seed(0)
    features = torch.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3).float()
    mask = torch.zeros(1, 2, 3, 3)
    mask[0, 0, 0, 0] = 1
    mask[0, 1, 2, 2] = 1

    sampled, labels = nef.sample_features_at_mask_locs(
        features,
        mask,
        nb_samples=1,
        return_label_ids=True,
    )

    assert sampled.shape == (1, 1, 2)
    assert labels.tolist() == [[0, 1]]
    assert sampled[0, 0].tolist() == [0.0, 8.0]


def test_one_hot_top_level_and_nn_class_subset():
    """Test one_hot class selection in top-level and NN APIs."""
    labels = torch.tensor([[[0, 1], [2, 1]]])
    top_level = ne.one_hot(labels, num_classes=3, class_list=[1, 2], non_spatial_dims=(0,))
    nn_level = nef.one_hot(labels.unsqueeze(1), num_classes=3, class_list=[1, 2])

    assert top_level.shape == (1, 2, 2, 2)
    assert torch.equal(top_level, nn_level)
    assert torch.equal(top_level[0, 0], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert torch.equal(top_level[0, 1], torch.tensor([[0.0, 0.0], [1.0, 0.0]]))


def test_filter_dim_removes_nan_slices():
    """Test that slices containing NaN are removed."""
    tensor = torch.tensor([
        [1.0, 2.0],
        [float('nan'), 3.0],
        [4.0, 5.0],
    ])
    result = ne.filter_dim(tensor, dim=0)
    assert result.shape == (2, 2)
    assert not torch.isnan(result).any()


def test_filter_dim_removes_inf_slices():
    """Test that slices containing Inf are removed."""
    tensor = torch.tensor([
        [1.0, 2.0],
        [float('inf'), 3.0],
        [4.0, float('-inf')],
        [6.0, 7.0],
    ])
    result = ne.filter_dim(tensor, dim=0)
    assert result.shape == (2, 2)
    assert not torch.isinf(result).any()


def test_filter_dim_removes_all_zero_slices():
    """Test that slices that are entirely zero are removed."""
    tensor = torch.tensor([
        [1.0, 2.0],
        [0.0, 0.0],
        [3.0, 4.0],
    ])
    result = ne.filter_dim(tensor, dim=0)
    assert result.shape == (2, 2)
    # Verify no all-zero rows remain
    assert not (result == 0).all(dim=1).any()


def test_filter_dim_preserves_valid_slices():
    """Test that valid slices are preserved with correct values."""
    tensor = torch.tensor([
        [1.0, 2.0],
        [float('nan'), 3.0],
        [4.0, 5.0],
    ])
    result = ne.filter_dim(tensor, dim=0)
    expected = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("dim", [0, 1])
def test_filter_dim_works_on_different_dims(dim):
    """Test that filtering works correctly along different dimensions."""
    # Create tensor with NaN in specific positions
    tensor = torch.ones(3, 4)
    if dim == 0:
        tensor[1, :] = float('nan')  # NaN in row 1
        result = ne.filter_dim(tensor, dim=0)
        assert result.shape == (2, 4)
    else:
        tensor[:, 2] = float('nan')  # NaN in column 2
        result = ne.filter_dim(tensor, dim=1)
        assert result.shape == (3, 3)


def test_parse_non_spatial_dims_rejects_non_leading_dims():
    """Non-spatial dims must be explicit leading dimensions."""
    with pytest.raises(AssertionError, match="leading contiguous"):
        ne.parse_non_spatial_dims((1,), tensor_ndim=4)


def test_ncc_supports_arbitrary_leading_non_spatial_dims():
    """Top-level NCC preserves arbitrary leading non-spatial dimensions."""
    tensor1 = torch.rand(2, 3, 4, 5, 16, 16)
    tensor2 = torch.rand(2, 3, 4, 5, 16, 16)

    score = ne.ncc(tensor1, tensor2, non_spatial_dims=(0, 1, 2, 3))

    assert score.shape == (2, 3, 4, 5)


def test_resample_supports_arbitrary_leading_non_spatial_dims():
    """Top-level resample preserves arbitrary leading non-spatial dimensions."""
    tensor = torch.rand(2, 3, 4, 5, 16, 16)

    result = ne.resample(tensor, scale_factor=0.5, non_spatial_dims=(0, 1, 2, 3))

    assert result.shape == (2, 3, 4, 5, 8, 8)
