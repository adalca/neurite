"""
Tests for `neurite` modules.
"""

import pytest
import torch
import neurite as ne


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
