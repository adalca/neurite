"""
pytest suite for BasicUNet.
"""

import pytest
import torch

# Replace `your_module` with the actual module path
# where BasicUNet is defined.
from neurite.pytorch.models import BasicUNet


@pytest.mark.parametrize(
    "ndim, spatial_size",
    [
        (1, (64,)),
        (2, (64, 64)),
        (3, (32, 32, 32)),
    ],
)
def test_basicunet_forward_shapes(ndim, spatial_size):
    """
    Test that BasicUNet returns a tensor of the correct shape.

    Parameters
    ----------
    ndim : int
        Dimensionality of the network (1, 2, or 3).
    spatial_size : tuple of ints
        Spatial dimensions for the input tensor.

    Returns
    -------
    None
    """
    batch_size = 2
    in_ch = 3
    out_ch = 5

    # Instantiate model with a small feature map
    model = BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[8, 16],
        norms=None,
        activations="relu",
    )
    model.eval()

    # Create a random input tensor
    x = torch.randn((batch_size, in_ch, *spatial_size))

    # Forward pass
    y = model(x)

    # Check type and shape
    assert isinstance(y, torch.Tensor)
    assert y.shape == (batch_size, out_ch, *spatial_size)


@pytest.mark.parametrize("residual", [True, False])
def test_basicunet_residual_option(residual):
    """
    Test that the residual_connections flag toggles skip connections
    without changing output shape.

    Parameters
    ----------
    residual : bool
        Whether to use residual connections.

    Returns
    -------
    None
    """
    ndim = 2
    in_ch = 1
    out_ch = 1
    size = (16, 16)

    model = BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[4, 8],
        residual_connections=residual,
    )
    model.eval()

    x = torch.randn((1, in_ch, *size))
    y = model(x)

    # Even without residuals, output dims stay the same
    assert y.shape == (1, out_ch, *size)


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_basicunet_upsample_modes(mode):
    """
    Test that different upsample_mode settings run without error
    and preserve spatial dimensions.

    Parameters
    ----------
    mode : str
        Upsampling mode to use (e.g., 'nearest', 'linear').

    Returns
    -------
    None
    """
    ndim = 2
    in_ch = 1
    out_ch = 1

    model = BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[8, 8],
        upsample_mode=mode,
    )
    model.eval()

    x = torch.randn((1, in_ch, 32, 32))
    y = model(x)

    assert y.shape == (1, out_ch, 32, 32)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)
def test_basicunet_forward_cuda():
    """
    Test that the model can be moved to CUDA and still produce correct output.

    Returns
    -------
    None
    """
    ndim = 2
    in_ch = 1
    out_ch = 1

    # Move model and data to GPU
    model = BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
    ).cuda()
    model.eval()

    x = torch.randn((1, in_ch, 16, 16), device="cuda")
    y = model(x)

    # Ensure output is on CUDA
    assert y.device.type == "cuda"
    assert y.shape == (1, out_ch, 16, 16)
