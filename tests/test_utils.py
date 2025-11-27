import pytest
import itertools

import torch
import neurite as ne
import neurite.nn.functional as nef


def test_base_gaussian_kernel_no_batch_channel():
    """Test that base gaussian_kernel returns only spatial dimensions."""
    # Use truncate=2 to get kernel_size=(5,5): 2*int(2*1+0.5)+1=5
    kernel = ne.gaussian_kernel(sigma=1.0, truncate=2, ndim=2)

    # Should have only spatial dimensions, no batch/channel
    assert kernel.shape == (5, 5)
    assert kernel.dim() == 2


def test_base_gaussian_kernel_sums_to_one():
    """Test that base gaussian_kernel is normalized."""
    kernel = ne.gaussian_kernel(sigma=2.5, truncate=1, ndim=3)

    # Should sum to 1
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)


def test_base_gaussian_kernel_different_ndims():
    """Test that base gaussian_kernel works for 1D, 2D, and 3D."""
    kernel_1d = ne.gaussian_kernel(sigma=1.0, truncate=2, ndim=1)
    kernel_2d = ne.gaussian_kernel(sigma=1.0, truncate=2, ndim=2)
    kernel_3d = ne.gaussian_kernel(sigma=1.0, truncate=2, ndim=3)

    assert kernel_1d.shape == (5,)
    assert kernel_2d.shape == (5, 5)
    assert kernel_3d.shape == (5, 5, 5)

    # All should sum to 1
    assert torch.allclose(kernel_1d.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(kernel_2d.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(kernel_3d.sum(), torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize(
    'grid_shape, expected_out_shape',
    (
        ((32,), (1, 32)),          # (ndim, *spatial)
        ((32, 32), (2, 32, 32)),
        ((32, 32, 32), (3, 32, 32, 32)),
    )
)
def test_grid_shape(grid_shape, expected_out_shape):

    # Must return stack with shape (ndim, *spatial)
    coord_grid = ne.volshape_to_ndgrid(grid_shape, stack=True)
    assert coord_grid.shape == expected_out_shape


@pytest.mark.parametrize(
    "grid_shape", [(32,), (32, 32), (32, 32, 32)]
)
def test_grid_normalized(grid_shape: tuple):
    """
    Test that when normalize=True and indexing='ij', every corner of the
    generated coordinate grid has values exactly -1 or 1.

    With (ndim, *spatial) format, indexing coord_grid[:, *corner] gives the
    coordinate vector at that spatial location.
    """
    coord_grid = ne.volshape_to_ndgrid(
        grid_shape, normalize=True, indexing='ij', stack=True
    )

    corners = tuple(itertools.product(*[[0, s - 1] for s in grid_shape]))

    expected_corner_values = tuple(
        tuple(-1 if c == 0 else 1 for c in corner)
        for corner in corners
    )

    # With (ndim, *spatial) format, use (slice(None),) + corner to index
    # This selects all ndim values at the given spatial corner
    corner_vals = torch.stack([
        coord_grid[(slice(None),) + corner] for corner in corners
    ], dim=0)

    expected_corner_values = torch.tensor(expected_corner_values)

    assert torch.all(corner_vals == expected_corner_values)


@pytest.mark.parametrize('n,mode', [
    (1, 'linear'),
    (2, 'bilinear'),
    (3, 'trilinear'),
])
def test_infer_linear_interpolation_mode(n, mode):
    assert ne.utils.infer_linear_interpolation_mode(n) == mode


def test_apply_bernoulli_mask_all_keep():
    t = torch.ones(10)
    masked = nef.apply_bernoulli_mask(t, p=1.0)
    assert masked.shape == t.shape
    assert torch.all(masked == 1)


def test_reduction():

    tensor = torch.randn(1, 1, 128, 128, 128)

    reduced_tensor = nef.reduce(tensor, reduction='mean', keepdims=False)

    assert reduced_tensor.shape == torch.Size([]), (
        f"Reduced tensor should be a scalar, got {reduced_tensor.shape}"
    )


@pytest.mark.parametrize(
    'prob, expected_range', (
        (0.1, (900, 1100)), (0.5, (4800, 5200))
    )
)
def test_random_flip(prob, expected_range):
    """
    Test random flip function to ensure it flips correctly along specified dimensions.

    Notes
    -----
    I calculated the expected ranges using the binomial distribution CDF `scipy.stats.binom`
    with n=10000 and p=prob. The ranges correspond to a 99% confidence interval. This test is
    expected to fail approximately 1% of the time due to the probabilistic nature of the function.
    """

    input_tensor = torch.randn(1, 1, 8, 8)
    flipped_tensor_gt = input_tensor.clone().flip([2])

    num_times_flipped = 0
    num_trials = 10_000

    for i in range(num_trials):
        flipped_tensor = nef.random_flip(2, input_tensor, prob=prob)
        if torch.equal(flipped_tensor, flipped_tensor_gt):
            num_times_flipped += 1

    assert expected_range[0] <= num_times_flipped <= expected_range[1], (
        f"Number of flips {num_times_flipped} out of {num_trials} trials with p={prob} "
        f"not within expected range {expected_range}. WARNING: This test is expected to fail ~1% of"
        "the time, due to the probabilistic nature of the function."
    )


def test_resize_scale_factor():
    """
    Test resizing constant tensor scales shape and preserves constant values.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    img = torch.ones((1, 2, 2), device=device)
    out = nef.resize(img, scale_factor=3)
    assert out.shape == (1, 6, 6)
    assert torch.allclose(out, torch.ones_like(out))


def test_resize_nearest():
    """
    Nearest neighbor resizing should match repeat_interleave for integer scale factors.
    """

    # Image to be resized
    img = torch.arange(16).view(1, 4, 4).float()

    # Resized image using neurite function
    resized_img = nef.resize(img, scale_factor=2, nearest=True).float()

    # Ground truth resized image using repeat_interleave
    img = torch.arange(16).view(1, 1, 4, 4).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    resized_img_gt = img.reshape(1, 8, 8).float()

    assert torch.allclose(resized_img, resized_img_gt, atol=1e-8)


def test_build_normalization_returns_module():
    """
    Regression test: build_normalization must return a module, not None.
    Bug: bare `return` statements returned None instead of the normalization.
    """
    # Test instantiated input
    norm = torch.nn.InstanceNorm2d(16)
    assert ne.utils.build_normalization(norm) is norm

    # Test uninstantiated class input
    result = ne.utils.build_normalization(torch.nn.BatchNorm2d, num_features=16)
    assert isinstance(result, torch.nn.BatchNorm2d)


def test_reduce_keepdim_parameter():
    """
    Regression test: reduce() must use keepdim (not keepdims) for PyTorch API.
    Bug: used `keepdims=` which is not a valid PyTorch parameter name.
    """
    tensor = torch.randn(2, 3, 4)
    result = ne.reduce(tensor, reduction='mean', dim=1, keepdims=True)
    assert result.shape == (2, 1, 4)
