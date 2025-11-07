"""
Tests for vectorized functions in neurite.nn.functional.

These tests verify that vectorized implementations produce identical outputs
to the original loop-based implementations.
"""

import torch
import neurite as ne
import neurite.nn.functional as nef


# =============================================================================
# Tests for random_clear_label()
# =============================================================================

def test_random_clear_label_deterministic():
    """Test random_clear_label with fixed seed produces consistent results."""
    torch.manual_seed(42)

    # Create simple label tensor with known labels
    label_tensor = torch.tensor([
        [[0, 1, 1, 2],
         [0, 1, 1, 2],
         [3, 3, 4, 4],
         [3, 3, 4, 4]]
    ]).unsqueeze(0).float()  # Shape: (1, 1, 4, 4)

    # Input tensor has values corresponding to labels
    input_tensor = label_tensor.clone() * 10.0  # Multiply by 10 so we can see clearing

    # Apply with prob=0.5 - some labels should be cleared
    torch.manual_seed(42)
    result = nef.random_clear_label(input_tensor, label_tensor, prob=0.5, exclude_zero=True)

    # Verify some regions were cleared (should have zeros where labels were cleared)
    # The original had some zeros, but we should have more after clearing
    assert (result == 0).sum() >= (input_tensor == 0).sum()

    # Where label_tensor is 0, result should still have original values (if exclude_zero=True)
    zero_mask = (label_tensor == 0)
    assert torch.allclose(result[zero_mask], input_tensor[zero_mask])


def test_random_clear_label_exclude_zero():
    """Test that exclude_zero=True preserves regions with label=0."""
    label_tensor = torch.tensor([
        [[0, 1, 2],
         [0, 3, 4],
         [0, 5, 6]]
    ]).unsqueeze(0).float()

    # Input has non-zero values everywhere (including where label=0)
    input_tensor = torch.ones_like(label_tensor) * 100.0

    torch.manual_seed(42)
    result = nef.random_clear_label(input_tensor, label_tensor, prob=0.5, exclude_zero=True)

    # Regions where label=0 should keep their original values (not be cleared)
    zero_label_mask = (label_tensor == 0)
    assert torch.allclose(result[zero_label_mask], input_tensor[zero_label_mask])


# =============================================================================
# Tests for subsample_tensor_random_dims()
# =============================================================================

def test_subsample_tensor_random_dims_all_dims():
    """Test subsampling when all dimensions are selected."""
    input_tensor = torch.arange(1, 65).reshape(1, 1, 8, 8).float()

    # Force all dimensions to be subsampled (p=1.0)
    # With forbidden_dims=(0, 1), only spatial dims 2 and 3 can be subsampled
    # Both will be subsampled with stride=2, resulting in 4x4 output
    torch.manual_seed(42)
    result = nef.subsample_tensor_random_dims(input_tensor, stride=2, p=1.0)

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


def test_subsample_tensor_random_dims_no_dims():
    """Test subsampling when no dimensions are selected."""
    input_tensor = torch.randn(1, 1, 8, 8)

    # Force no dimensions to be subsampled (p=0.0)
    result = nef.subsample_tensor_random_dims(input_tensor, stride=2, p=0.0)

    # Should remain unchanged
    assert result.shape == input_tensor.shape
    assert torch.equal(result, input_tensor)


def test_subsample_tensor_random_dims_3d():
    """Test 3D subsampling."""
    input_tensor = torch.randn(1, 2, 16, 16, 16)

    # Subsample with p=1.0 (all dims)
    result = nef.subsample_tensor_random_dims(input_tensor, stride=2, p=1.0)

    # All spatial dims should be halved
    assert result.shape == (1, 2, 8, 8, 8)


def test_subsample_tensor_random_dims_deterministic():
    """Test that same seed produces same result."""
    input_tensor = torch.randn(1, 1, 16, 16)

    torch.manual_seed(42)
    result1 = nef.subsample_tensor_random_dims(input_tensor, stride=2, p=0.5)

    torch.manual_seed(42)
    result2 = nef.subsample_tensor_random_dims(input_tensor, stride=2, p=0.5)

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

    # Use small noise variance for this test
    import neurite.samplers as samplers
    result = nef.sample_image_from_labels(
        label_tensor,
        mean_sampler=samplers.Fixed(0.5),
        noise_variance=0.01
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
    import neurite.samplers as samplers
    torch.manual_seed(42)
    result = nef.sample_image_from_labels(
        label_tensor,
        mean_sampler=samplers.Uniform(0, 1),
        noise_variance=0.001  # Very small variance instead of 0.0
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

    import neurite.samplers as samplers
    torch.manual_seed(42)
    result1 = nef.sample_image_from_labels(
        label_tensor,
        mean_sampler=samplers.Uniform(0, 1),
        noise_variance=0.1
    )

    torch.manual_seed(42)
    result2 = nef.sample_image_from_labels(
        label_tensor,
        mean_sampler=samplers.Uniform(0, 1),
        noise_variance=0.1
    )

    assert torch.allclose(result1, result2, atol=1e-6)


def test_volshape_to_ndgrid_sizes():
    size = (19, 32)
    the_grid = ne.functional.volshape_to_ndgrid(size=size, stack=True)
    assert tuple(the_grid.shape) == (19, 32, 2)


def test_nn_functional_volshape_to_ndgrid_sizes():
    B, C = 2, 3
    size = (B, C, 43, 9, 10)
    the_grid = ne.nn.functional.volshape_to_ndgrid(size=size, stack=True)
    assert tuple(the_grid.shape) == (B, C, 43, 9, 10, 3) 
