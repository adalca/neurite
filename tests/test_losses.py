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


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0, -1.0])
def test_ncc_scale_invariance(alpha):
    """Test that NCC is scale-invariant: ncc(x, alpha*x) = 1 for any alpha != 0."""
    torch.manual_seed(42)
    t1 = torch.rand(2, 1, 64, 64)
    t2 = alpha * t1
    score = nef.ncc(t1, t2, reduction=None)
    assert torch.allclose(score, torch.ones_like(score), atol=1e-5), (
        f"NCC(x, {alpha}*x) should be 1, got {score.mean().item():.6f}"
    )


def test_ncc_window_size():
    """Test NCC with different window sizes."""
    t1 = torch.rand(2, 1, 64, 64)
    t2 = torch.rand(2, 1, 64, 64)

    # Should work with different window sizes
    for win in [3, 5, 9, 11]:
        score = nef.ncc(t1, t2, window_size=win)
        assert score.shape == (1, 1), f"Expected (1, 1), got {score.shape} for window_size={win}"

    # Should work with per-dimension window sizes
    score = nef.ncc(t1, t2, window_size=[5, 9])
    assert score.shape == (1, 1), f"Expected (1, 1), got {score.shape} for per-dim window"


def test_ncc_module():
    """Test NCC module wrapper."""
    t1 = torch.rand(4, 3, 64, 64)

    # Test with reduction
    ncc_module = ne.nn.modules.NCC(reduction='mean')
    score = ncc_module(t1, t1)
    assert score.shape == (1, 1), f"Expected (1, 1), got {score.shape}"
    assert torch.allclose(score, torch.ones(1, 1), atol=1e-6), (
        "NCC module for identical tensors should be 1.0"
    )

    # Test without reduction
    ncc_module_no_red = ne.nn.modules.NCC(reduction=None)
    score = ncc_module_no_red(t1, t1)
    assert score.shape == (4, 3), f"Expected (4, 3), got {score.shape}"


def test_base_spatial_gradient():
    """Test the shape-agnostic base spatial_gradient function."""
    # Test with no non-spatial dims (all spatial)
    t = torch.rand(64, 64)
    grads = ne.spatial_gradient(t, non_spatial_dims=None)
    assert len(grads) == 2, f"Expected 2 gradients for 2D, got {len(grads)}"
    assert grads[0].shape == (63, 64), f"Expected (63, 64), got {grads[0].shape}"
    assert grads[1].shape == (64, 63), f"Expected (64, 63), got {grads[1].shape}"

    # Test with B, C dims
    t = torch.rand(2, 3, 32, 32, 32)
    grads = ne.spatial_gradient(t, non_spatial_dims=(0, 1))
    assert len(grads) == 3, f"Expected 3 gradients for 3D, got {len(grads)}"
    assert grads[0].shape == (2, 3, 31, 32, 32), (
        f"Expected (2, 3, 31, 32, 32), got {grads[0].shape}"
    )


def test_spatial_gradient_values():
    """Test that spatial gradient computes correct finite differences."""
    # 1D case: [0, 1, 2, 3, 4] -> diffs should be [1, 1, 1, 1]
    t = torch.arange(5).float()
    grads = ne.spatial_gradient(t, non_spatial_dims=None)
    expected = torch.ones(4)
    assert torch.allclose(grads[0], expected), (
        f"Expected {expected.tolist()}, got {grads[0].tolist()}"
    )


def test_spatial_gradient_magnitudes():
    """Test spatial gradient with known 2D grid values."""
    x = torch.arange(100).reshape(10, 10).float()

    gradients = ne.spatial_gradient(x, non_spatial_dims=None)

    expected_dim0 = torch.ones(9, 10) * 10  # stepping down rows jumps by 10
    expected_dim1 = torch.ones(10, 9)        # stepping across cols jumps by 1

    assert torch.allclose(gradients[0], expected_dim0)
    assert torch.allclose(gradients[1], expected_dim1)


def test_spatial_gradient_constant():
    """Test that constant fields have zero gradient."""
    t = torch.ones(2, 3, 32, 32)
    loss = nef.spatial_gradient(t, penalty='l2')
    assert loss.item() == 0.0, f"Expected 0 for constant field, got {loss.item()}"


def test_spatial_gradient_penalties():
    """Test L1 and L2 penalties produce different results."""
    torch.manual_seed(42)
    t = torch.rand(2, 3, 64, 64)

    l1_loss = nef.spatial_gradient(t, penalty='l1')
    l2_loss = nef.spatial_gradient(t, penalty='l2')

    # L1 and L2 should produce different values
    assert not torch.allclose(l1_loss, l2_loss), (
        "L1 and L2 penalties should produce different values"
    )

    # Both should be non-negative
    assert l1_loss >= 0, f"L1 loss should be non-negative, got {l1_loss}"
    assert l2_loss >= 0, f"L2 loss should be non-negative, got {l2_loss}"


def test_spatial_gradient_module():
    """Test SpatialGradient module wrapper."""
    t = torch.rand(2, 3, 64, 64)

    # Test with default settings
    grad_module = ne.nn.modules.SpatialGradient(penalty='l2')
    loss = grad_module(t)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    # Test with L1 penalty
    grad_module_l1 = ne.nn.modules.SpatialGradient(penalty='l1')
    loss_l1 = grad_module_l1(t)
    assert loss_l1.ndim == 0, f"Expected scalar, got shape {loss_l1.shape}"
