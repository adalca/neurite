"""
Module for testing the losses of `neurite`. To be ran with `pytest`.
"""
import pytest
import torch
import neurite as ne


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
    assert ne.utils.utils.mse(t1, t2).item() == 1.0


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
        ne.nn.functional.dice(seg, seg, reduction=None)
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
    result = ne.utils.utils.dice(seg, seg)

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
    dice = ne.utils.utils.dice(seg1, seg2, reduction=None)

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
    dice = ne.utils.utils.dice(seg1, seg2, reduction=None)

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
    dice = ne.utils.utils.dice(*segs, reduction=None)

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
    dice = ne.losses.Dice(reduction=None)

    # Calculate the dice score
    result = dice(seg, seg)

    # The expected value for the dice score for identical inputs
    expected = torch.tensor(1.0)

    # Allow a small tolerance because of smoothing constants
    assert torch.allclose(result, expected, atol=1e-6), (
        "Dice for identical inputs should be close to 1."
    )


def test_log_dice(log_probabilities):
    """
    Test the log-dice score with log-probabilities
    """

    log_probs = log_probabilities

    # Compute dice on log between the same tensor
    log_dice_score = ne.utils.log_dice(log_probs, log_probs)

    # The expected log dice (still in the log domain) should be zero
    expected = torch.tensor(0.0)

    assert torch.allclose(log_dice_score, expected, atol=1e-6), (
        f"Log dice for identical tensors should be close to 1. Got {log_dice_score}"
    )


@pytest.mark.parametrize(
        'n_segs', (2, 5)
)
def test_multiple_log_dice(n_segs, log_probabilities):
    """
    Test the log-dice score with multiple log-probabilities
    """

    log_probs = [log_probabilities] * n_segs

    # Compute dice on log between the same tensor
    log_dice_score = ne.utils.log_dice(*log_probs, reduction='mean')

    # The expected log dice (still in the log domain) should be zero
    expected = torch.tensor(0.0)

    assert torch.allclose(log_dice_score, expected, atol=1e-4), (
        f"Log dice for identical tensors should be close to 1. Got {log_dice_score}"
    )
