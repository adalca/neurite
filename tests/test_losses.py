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


def test_dice_shapes():
    """
    Test the different shapes that the dice score can potentially handle

    The dice score should at least be able to handle 1D, 2D, and 3D tensors.
    """

    # Dice is able to handle multiple dimensions on the fly
    dice = ne.losses.Dice(reduction=None)

    # Init tensors with batch and channel dims: (B, C, *spatial)
    tensor_1D = torch.ones(1, 1, 8)
    tensor_2D = torch.ones(1, 1, 8, 8)
    tensor_3D = torch.ones(1, 1, 8, 8, 8)

    # Test the 1D case
    try:
        dice(tensor_1D, tensor_1D)
    except Exception as e:
        pytest.fail(f"Dice failed for 1D tensor: {e}")

    # Test the 2D case
    try:
        dice(tensor_2D, tensor_2D)
    except Exception as e:
        pytest.fail(f"Dice failed for 2D tensor: {e}")

    # Test the 3D case
    try:
        dice(tensor_3D, tensor_3D)
    except Exception as e:
        pytest.fail(f"Dice failed for 3D tensor: {e}")


def test_dice_identical():
    """
    Test Dice module with identical inputs.

    When both segmentation tensors are identical, the Dice coefficient
    should be nearly 1 (accounting for smoothing).
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


@pytest.mark.parametrize("spatial_dims", [1, 2, 3])
def test_dice_nonidentical(spatial_dims):
    """
    Test the Dice {1, 2, 3} dim dice score with non-identical inputs.
    """

    # Pick some sizes:
    batch, channels, spatial = 2, 3, 32
    spatial = (batch, channels) + (spatial,) * spatial_dims

    # Make uniformly distributed binary tensors
    seg1 = torch.randint(0, 2, (spatial)).float()
    seg2 = 1 - seg1

    # Compute the dice score
    dice = ne.utils.utils.dice(
        seg1=seg1,
        seg2=seg2,
        reduction=None
    )

    # The expected value for the dice of uniformly distributed binary tensors should be 0
    expected = torch.zeros(batch, channels)

    # Allow for a small tolderance because of smoothing and stochasticity
    assert torch.allclose(dice, expected, atol=1e-5), (
        "Dice for uniformly distributed binary tensors should be close to 0.5"
    )


def test_log_dice(log_probabilities):
    """
    Test the log-dice score with log-probabilities
    """

    log_probs = log_probabilities

    # Compute dice on log between the same tensor
    log_dice_score = ne.utils.log_dice(
        seg1=log_probs,
        seg2=log_probs,
    )

    # The expected log dice (still in the log domain) should be zero
    expected = torch.tensor(0.0)

    assert torch.allclose(log_dice_score, expected, atol=1e-6), (
        f"Log dice for identical tensors should be close to 1. Got {log_dice_score}"
    )

