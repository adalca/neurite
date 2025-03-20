"""
Module for testing the losses of `neurite`. To be ran with `pytest`
"""
import pytest
import torch
import neurite as ne


def test_dice_shapes():
    """
    Test the different shapes that the dice score can potentially handle

    The dice score should at least be able to handle 1D, 2D, and 3D tensors.
    """

    # Dice is able to handle multiple dimensions on the fly
    dice = ne.pytorch.losses.Dice(reduction=None)

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
    seg = torch.ones((1, 1, 8, 8))

    # Initialize the loss
    dice = ne.pytorch.losses.Dice(reduction=None)

    # Calculate the dice score
    result = dice(seg, seg)

    # The expected value for the dice score for identical inputs
    expected = torch.tensor(1.0)

    # Allow a small tolerance because of smoothing constants
    assert torch.allclose(result, expected, atol=1e-6), (
        "Dice for identical inputs should be close to 1."
    )


def test_dice_nonidentical():
    """
    Test the Dice module with non-identical inputs.

    The dice for uniformly distributed binary tensors should be close to 0.5.
    """

    # Make uniformly distributed binary tensors
    seg1 = torch.randint(2, (1, 1, 128, 128)).float()
    seg2 = torch.randint(2, (1, 1, 128, 128)).float()

    # Initialize the dice score
    dice = ne.pytorch.losses.Dice(reduction=None)

    # Compute the dice score
    result = dice(seg1, seg2)

    # The expected value for the dice of uniformly distributed binary tensors should be ~0.5
    expected = torch.tensor(0.5)

    # Allow for a small tolderance because of smoothing and stochasticity
    assert torch.allclose(result, expected, atol=1e-2), (
        "Dice for uniformly distributed binary tensors should be close to 0.5"
    )
