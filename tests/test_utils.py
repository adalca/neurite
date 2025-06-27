import pytest
import itertools

import torch
import neurite as ne


def test_soft_quantize_constant_input_same_output():
    """A constant tensor remains unchanged after soft quantization."""

    const_val = 3.14
    input_tensor = torch.full((2, 3), const_val)
    output_tensor = ne.utils.utils.soft_quantize(input_tensor.clone(), nb_bins=10, softness=2.0)

    assert torch.allclose(output_tensor, input_tensor)


@pytest.mark.parametrize("softness", [0.5, 1.0, 2.0])
def test_soft_quantize_monotonic_increasing(softness):
    """
    For a strictly increasing 1D input, the quantized output should be non-decreasing.
    """

    input_tensor = torch.linspace(0.0, 1.0, steps=5)

    output_tensor = ne.utils.utils.soft_quantize(
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

    output_tensor = ne.utils.utils.soft_quantize(
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
    For nchannels>1, kernel shape should be
    [nchannels, nchannels, *spatial_dims], and each channel
    kernel should be identical.
    """

    nchannels = 3
    kernel_size, sigma, ndim = 5, 1.0, 3
    kernel = ne.utils.utils.gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        ndim=ndim,
        nchannels=nchannels
    )

    # shape check
    expected = (nchannels, nchannels) + (kernel_size,) * ndim
    assert kernel.shape == expected

    # spatial kernels along the diagonal should match
    spatial_00 = kernel[0, 0]
    spatial_11 = kernel[1, 1]

    assert torch.allclose(spatial_00, spatial_11)


def test_subsample_tensor_magnitudes():
    """
    Ensure subsampling produces expected magnitudes/skips.
    """

    input_tensor = torch.arange(25).view(1, 1, 5, 5).float()
    subsampled_gt = torch.tensor([0, 2, 4, 10, 12, 14, 20, 22, 24]).view(1, 1, 3, 3).float()

    subsampled_tensor = ne.utils.utils.subsample(input_tensor, stride=2)

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

    ne.utils.utils.subsample(
        input_tensor,
        stride=stride,
        subsampling_dimension=subsampling_dimension,
    )


@pytest.mark.parametrize(
        'grid_shape, expected_out_shape',
        (
            ((32,), (1, 32, 1)),
            ((32, 32), (1, 32, 32, 2)),
            ((32, 32, 32), (1, 32, 32, 32, 3)),
        )
)
def test_grid_shape(grid_shape, expected_out_shape):

    coord_grid = ne.utils.utils.grid(grid_shape)
    assert coord_grid.shape == expected_out_shape


@pytest.mark.parametrize(
        "grid_shape", [(32,), (32, 32), (32, 32, 32)]
)
def test_grid_normalized(grid_shape: tuple):
    """
    Test that when normalize=True and indexing='ij', every corner of the
    generated coordinate grid has values exactly -1 or 1.
    """

    coord_grid = ne.utils.utils.grid(grid_shape, normalize=True, indexing='ij')
    corners = tuple(itertools.product(*[[0, s - 1] for s in grid_shape]))

    expected_corner_values = tuple(
        tuple(-1 if c == 0 else 1 for c in corner)
        for corner in corners
    )

    corner_vals = torch.stack([
        coord_grid[(0, *corner)] for corner in corners
    ], dim=0)

    expected_corner_values = torch.tensor(expected_corner_values)

    assert torch.all(corner_vals == expected_corner_values)


def test_logistic_midpoint():
    out = ne.utils.utils.logistic(torch.tensor(0.0))
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
    masked = ne.utils.utils.apply_bernoulli_mask(t, p=1.0)
    assert masked.shape == t.shape
    assert torch.all(masked == 1)
