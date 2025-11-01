import pytest
import itertools

import torch
import neurite as ne
import neurite.nn.functional as nef


def test_soft_quantize_constant_input_same_output():
    """A constant tensor remains unchanged after soft quantization."""

    const_val = 3.14
    input_tensor = torch.full((2, 3), const_val)
    output_tensor = nef.soft_quantize(input_tensor.clone(), nb_bins=10, softness=2.0)

    assert torch.allclose(output_tensor, input_tensor)


@pytest.mark.parametrize("softness", [0.5, 1.0, 2.0])
def test_soft_quantize_monotonic_increasing(softness):
    """
    For a strictly increasing 1D input, the quantized output should be non-decreasing.
    """

    input_tensor = torch.linspace(0.0, 1.0, steps=5)

    output_tensor = nef.soft_quantize(
        input_tensor.clone(),
        nb_bins=5,
        softness=softness
    )

    assert torch.all(output_tensor[:-1] <= output_tensor[1:])


def test_soft_quantize_clipping():
    """
    When specifying min_clip and max_clip, outputs must lie
    within [min_clip, max_clip].
    """

    input_tensor = torch.randn(1, 1, 16, 16)

    output_tensor = nef.soft_quantize(
        input_tensor.clone(),
        nb_bins=3,
        softness=1.0,
        min_clip=0.0,
        max_clip=1.0
    )

    assert torch.all(output_tensor >= 0.0)
    assert torch.all(output_tensor <= 1.0)


def test_create_gaussian_kernel_sums_to_one():
    """Make sure kernel is normalized (sums to 1)"""

    kernel = ne.utils.utils.gaussian_kernel(
        kernel_size=7,
        sigma=2.5,
        ndim=2,
        nchannels=1
    )

    total = kernel.sum()

    assert torch.allclose(total, torch.tensor(1.0), atol=1e-6)


def test_create_gaussian_kernel_shape_and_symmetry():
    """
    For nchannels>1, kernel shape should be nchannels, 1, *spatial_dims] for depthwise convolution,
    and each channel kernel should be identical.
    """

    nchannels = 3
    kernel_size, sigma, ndim = 5, 1.0, 3
    kernel = ne.utils.utils.gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        ndim=ndim,
        nchannels=nchannels
    )

    # shape check for depthwise convolution
    expected = (nchannels, 1) + (kernel_size,) * ndim
    assert kernel.shape == expected

    # spatial kernels for each channel should be identical
    spatial_0 = kernel[0, 0]
    spatial_1 = kernel[1, 0]

    assert torch.allclose(spatial_0, spatial_1)


def test_subsample_tensor_magnitudes():
    """
    Ensure subsampling produces expected magnitudes/skips.
    """

    input_tensor = torch.arange(25).view(1, 1, 5, 5).float()
    subsampled_gt = torch.tensor([0, 2, 4, 10, 12, 14, 20, 22, 24]).view(1, 1, 3, 3).float()

    subsampled_tensor = nef.subsample(input_tensor, stride=2)

    torch.allclose(subsampled_tensor, subsampled_gt)


@pytest.mark.parametrize(
    "stride, subsampling_dimension, input_tensor",
    [
        (2, None, torch.randn(1, 1, 32, 32, 32)),
        ([2], [0], torch.randn(1, 1, 32)),
        ((2, 4), (0, 1), torch.randn(1, 1, 32, 32)),
        ((2, 4, 6), [0, 1, 2], torch.randn(1, 1, 32, 32, 32)),
    ],
)
def test_subsample_tensor_strides(
    stride,
    subsampling_dimension,
    input_tensor,
):
    """
    Ensure subsampling produces expected magnitudes/skips.
    """

    # input_tensor = torch.randn(1, 1, *[32] * len(stride))

    nef.subsample(
        input_tensor,
        stride=stride,
        subsampling_dimension=subsampling_dimension,
    )


@pytest.mark.parametrize(
    'grid_shape, expected_out_shape',
    (
        ((32,), (32, 1)),
        ((32, 32), (32, 32, 2)),
        ((32, 32, 32), (32, 32, 32, 3)),
    )
)
def test_grid_shape(grid_shape, expected_out_shape):

    # Must return stack!
    coord_grid = nef.volshape_to_ndgrid(grid_shape, stack=True)
    assert coord_grid.shape == expected_out_shape


@pytest.mark.parametrize(
    "grid_shape", [(32,), (32, 32), (32, 32, 32)]
)
def test_grid_normalized(grid_shape: tuple):
    """
    Test that when normalize=True and indexing='ij', every corner of the
    generated coordinate grid has values exactly -1 or 1.
    """

    coord_grid = ne.nn.functional.volshape_to_ndgrid(
        grid_shape, normalize=True, indexing='ij', stack=True
    )

    corners = tuple(itertools.product(*[[0, s - 1] for s in grid_shape]))

    expected_corner_values = tuple(
        tuple(-1 if c == 0 else 1 for c in corner)
        for corner in corners
    )

    corner_vals = torch.stack([
        coord_grid[corner] for corner in corners
    ], dim=0)

    expected_corner_values = torch.tensor(expected_corner_values)

    assert torch.all(corner_vals == expected_corner_values)


def test_logistic_midpoint():
    out = nef.logistic(torch.tensor(0.0))
    assert torch.allclose(out, torch.tensor(0.5))


@pytest.mark.parametrize('n,mode', [
    (1, 'linear'),
    (2, 'bilinear'),
    (3, 'trilinear'),
])
def test_infer_linear_interpolation_mode(n, mode):
    assert ne.utils.utils.infer_linear_interpolation_mode(n) == mode


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
