"""
Tests for `neurite` modules.
"""

import pytest
import torch
import neurite as ne
import neurite.nn.functional as nef


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('spatial, options, downscale, output_shape', [
    ((8,), {}, (0.5,), (8,)),
    ((8, 8), {}, (0.5, 0.5), (8, 8)),
    ((8, 8, 8), {}, (0.5, 0.5, 0.5), (8, 8, 8)),
    ((9, 11), {}, (0.5, 0.5), (9, 11)),
    ((8, 8), {'resample_dimension': 2}, (0.5, 1), (8, 8)),
    ((8, 8), {'resample_dimension': -1}, (1, 0.5), (8, 8)),
    ((8, 8, 8), {'resample_dimension': (2, 4)}, (0.5, 1, 0.5), (8, 8, 8)),
    ((8, 8), {'downsample_stride': (2, 4), 'upsample_scale_factor': (2, 4)},
     (0.5, 0.25), (8, 8)),
    ((8, 8), {'upsample_scale_factor': 6}, (0.5, 0.5), (24, 24)),
    ((8, 8), {'shape': (10, 12)}, (0.5, 0.5), (10, 12)),
])
def test_resample_voxel_dimensions(device, spatial, options, downscale, output_shape):
    """Check resolution degradation, selected axes, output sizes, and gradients."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA device not available')

    tensor = torch.randn(2, 3, *spatial, device=device, requires_grad=True)
    layer = ne.nn.modules.ResampleVoxelDimensions(**options)
    result = layer(tensor)

    # Compare with explicit interpolation factors, including untouched spatial axes.
    mode = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}[len(spatial)]
    reduced = torch.nn.functional.interpolate(tensor, scale_factor=downscale, mode=mode)
    expected = torch.nn.functional.interpolate(reduced, size=output_shape, mode=mode)
    torch.testing.assert_close(result, expected)
    assert result.shape == (2, 3, *output_shape)
    assert result.device == tensor.device
    result.sum().backward()
    assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize('options', [
    {'resample_dimension': 0},
    {'resample_dimension': 4},
    {'downsample_stride': (2,)},
    {'upsample_scale_factor': (2,)},
    {'downsample_stride': 0},
    {'upsample_scale_factor': -1},
])
def test_resample_voxel_dimensions_invalid_options(options):
    """Reject nonspatial axes, incorrect factor lengths, and nonpositive factors."""
    layer = ne.nn.modules.ResampleVoxelDimensions(**options)
    with pytest.raises(ValueError):
        layer(torch.ones(1, 1, 8, 8))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_norm_instance(ndim: int):
    """
    Test instance normalization for properly normalized outputs (on a per-batch and per-channel
    basis)
    """

    # Create input with nonzero mean and non-unit variance of shape (B, C, *spatial)
    spatial = [64] * ndim
    x = torch.randn(8, 4, *spatial) * 5 + 10

    # Initialize the normalization layer
    norm_layer = ne.utils.build_normalization(
        normalization_type='instance',
        ndim=ndim,
        num_features=4,
        affine=False
    )

    # Use current batch statistics, not the stored ones
    norm_layer.train()

    # Compute the normalization
    y = norm_layer(x)

    # Statistics are computed per instance per channel over spatial dims
    dimensions = [-1, -2, -3]
    y_mean = y.mean(dim=dimensions[0:ndim])
    y_var = y.var(dim=dimensions[0:ndim], unbiased=False)

    # Set the tolerance for the maximum difference from expectations
    tol = 1e-4

    # Ensure means are close to zero
    if not torch.allclose(y_mean, torch.zeros_like(y_mean), atol=tol):
        pytest.fail(
            f"Some means of instance normalization are not within +/- {tol} of zero. Got\n{y_mean}"
        )

    # Ensure variances are close to one
    if not torch.allclose(y_var, torch.ones_like(y_var), atol=tol):
        pytest.fail(
            f"Some variances of instance normalization are not within +/- {tol} of one. Got {y_var}"
        )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_norm_batch(ndim: int):
    """
    Test batch normalization for properly normalized outputs (for each channel over all batch and
    spatial dimensions)
    """

    # Create input with nonzero mean and non-unit variance of shape (B, C, *spatial)
    spatial = [64] * ndim
    x = torch.randn(4, 8, *spatial) * 5 + 10

    # Initialize the normalization layer
    norm_layer = ne.utils.build_normalization(
        normalization_type='batch',
        ndim=ndim,
        num_features=8,
        affine=False,
    )

    # Use current batch statistics, not the stored ones
    norm_layer.train()

    # Compute the normalization
    y = norm_layer(x)

    # Statistics are computed per channel over all batch and spatial dims
    dimensions = [[0, 2], [0, 2, 3], [0, 2, 3, 4]]
    y_mean = y.mean(dim=dimensions[ndim - 1])
    y_var = y.var(dim=dimensions[ndim - 1], unbiased=False)

    # Set the tolerance for the maximum difference from expectations
    tol = 1e-4

    # Ensure means are close to zero
    if not torch.allclose(y_mean, torch.zeros_like(y_mean), atol=tol):
        pytest.fail(
            f"Some means of batch normalization are not within +/- {tol} of zero. Got\n{y_mean}"
        )

    # Ensure variances are close to one
    if not torch.allclose(y_var, torch.ones_like(y_var), atol=tol):
        pytest.fail(
            f"Some variances of batch normalization are not within +/- {tol} of one. Got {y_var}"
        )
