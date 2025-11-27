"""
Module for testing the losses of `neurite`. To be ran with `pytest`.
"""
import pytest
import torch
import neurite as ne
import neurite.nn.functional as nef


def test_base_dice():
    seg1 = torch.randn(1, 1, 128, 128).sigmoid()
    seg2 = seg1.clone()
    dice_score = ne.dice(seg1, seg2)

    # Assert it's a scalar tensor
    is_tensor = isinstance(dice_score, torch.Tensor)
    is_no_dim = dice_score.ndim == 0
    is_scalar = True is is_tensor & is_no_dim
    assert is_scalar, f"ne.dice() must return a scalar by default. Got: {dice_score}"


@pytest.fixture
def log_probabilities():
    # Initialize the logits from a normal dist
    logits = torch.randn(1, 1, 128, 128)

    # Convert logits to valid probabilities
    probabilities = torch.nn.functional.sigmoid(logits)

    # Turn into log probabilities
    log_probabilities = probabilities.log()

    return log_probabilities


def test_mse_simple():
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([2.0, 3.0])
    assert nef.mse(t1, t2).item() == 1.0


@pytest.mark.parametrize('spatial_dims', [1, 2, 3])
def test_dice_shapes(spatial_dims):
    """
    Test the different shapes that the dice score can potentially handle

    The dice score should at least be able to handle 1D, 2D, and 3D tensors.
    """

    # Pick some sizes
    batch, channels, spatial = 2, 3, 32
    spatial = (batch, channels) + (spatial,) * spatial_dims

    # Initialize the seg
    seg = torch.randint(0, 2, spatial).float()

    try:
        # Dice is able to handle multiple dimensions on the fly
        nef.dice(seg, seg, reduction=None)
    except Exception as e:
        pytest.fail(f"Dice failed for tensor with {spatial_dims} spatial dims: {e}")


def test_dice_identical():
    """
    Test Dice module with identical inputs.

    When both segmentation tensors are identical, the Dice coefficient
    should be nearly 1 (accounting for smoothing).
    """

    # Create an example tensor of shape (B, C, H, W)
    seg = torch.ones((4, 3, 8, 8))

    # Calculate the dice score
    result = nef.dice(seg, seg)

    # The expected value for the dice score for identical inputs
    expected = torch.tensor(1.0)

    # Allow a small tolerance because of smoothing constants
    assert torch.allclose(result, expected, atol=1e-6), (
        "Dice for identical inputs should be close to 1."
    )


def test_dice_nonidentical():
    """
    Ensure dice returns zeros for opposite inputs.
    """

    # Make uniformly distributed binary tensors
    seg1 = torch.randint(0, 2, (4, 3, 256, 256, 256)).float()
    seg2 = torch.randint(0, 2, (4, 3, 256, 256, 256)).float()

    # Compute the dice score
    dice = nef.dice(seg1, seg2, reduction=None)

    # The expected value for the dice of uniformly distributed binary tensors should be 0.5
    expected = torch.ones(4, 3) * 0.5

    # Allow for a small tolderance because of smoothing and stochasticity
    assert torch.allclose(dice, expected, atol=1e-3), (
        "Dice for uniformly distributed binary tensors should be close to 0.5"
    )


def test_dice_opposite():
    """
    Ensure dice returns zeros for opposite inputs.
    """

    # Make uniformly distributed binary tensors
    seg1 = torch.randint(0, 2, (4, 3, 128, 128, 128)).float()
    seg2 = 1 - seg1

    # Compute the dice score
    dice = nef.dice(seg1, seg2, reduction=None)

    # The expected value for the dice of uniformly distributed binary tensors should be 0
    expected = torch.zeros(4, 3)

    # Allow for a small tolderance because of smoothing and stochasticity
    assert torch.allclose(dice, expected, atol=1e-5), (
        "Dice for uniformly distributed binary tensors should be close to 0.5"
    )


def test_multiple_dice_nonidentical():
    """
    Ensure dice returns zeros for opposite inputs.
    """

    # Make uniformly distributed binary tensors
    segs = [torch.randint(0, 2, (4, 3, 256, 256, 256)).float() for _ in range(3)]

    # Compute the dice score
    dice = nef.dice(*segs, reduction=None)

    # The expected value for the dice of 3 uniformly distributed binary tensors should be 0.25
    expected = torch.ones(4, 3) * 0.25

    # Allow for a small tolderance because of smoothing and stochasticity
    assert torch.allclose(dice, expected, atol=1e-3), (
        "Dice for uniformly distributed binary tensors should be close to 0.5"
    )


def test_dice_wrapper():
    """
    Test Dice module with identical inputs.
    """

    # Create an example tensor of shape (B, C, H, W)
    seg = torch.ones((4, 3, 8, 8))

    # Initialize the loss
    dice = ne.nn.modules.Dice(reduction=None)

    # Calculate the dice score
    result = dice(seg, seg)

    # The expected value for the dice score for identical inputs
    expected = torch.tensor(1.0)

    # Allow a small tolerance because of smoothing constants
    assert torch.allclose(result, expected, atol=1e-6), (
        "Dice for identical inputs should be close to 1."
    )


def test_base_ncc():
    """Test the shape-agnostic base NCC function returns correct shapes."""
    # Test with no non-spatial dims (all spatial) -> scalar
    t1 = torch.rand(64, 64)
    t2 = torch.rand(64, 64)
    score = ne.ncc(t1, t2, non_spatial_dims=None)
    assert score.ndim == 0, f"Expected scalar, got shape {score.shape}"

    # Test with batch dim only
    t1 = torch.rand(4, 64, 64)
    t2 = torch.rand(4, 64, 64)
    score = ne.ncc(t1, t2, non_spatial_dims=(0,))
    assert score.shape == (4,), f"Expected (4,), got {score.shape}"

    # Test with B, C dims
    t1 = torch.rand(2, 3, 64, 64)
    t2 = torch.rand(2, 3, 64, 64)
    score = ne.ncc(t1, t2, non_spatial_dims=(0, 1))
    assert score.shape == (2, 3), f"Expected (2, 3), got {score.shape}"


def test_ncc_identical():
    """Test NCC returns ~1 for identical tensors."""
    t1 = torch.rand(4, 3, 64, 64)

    score = nef.ncc(t1, t1, reduction=None)

    expected = torch.ones(4, 3)
    assert torch.allclose(score, expected, atol=1e-6), (
        f"NCC for identical tensors should be 1.0, got {score.mean().item()}"
    )


def test_ncc_different():
    """Test NCC returns values in [0, 1] for different random tensors."""
    torch.manual_seed(42)
    t1 = torch.rand(4, 3, 64, 64)
    t2 = torch.rand(4, 3, 64, 64)

    score = nef.ncc(t1, t2, reduction=None)

    # Score should be in [0, 1]
    assert (score >= 0).all() and (score <= 1).all(), (
        f"NCC values should be in [0, 1], got min={score.min()}, max={score.max()}"
    )
