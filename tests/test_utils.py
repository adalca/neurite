import pytest
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

    kernel = ne.utils.utils.create_gaussian_kernel(
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
    kernel = ne.utils.utils.create_gaussian_kernel(
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


