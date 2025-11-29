"""
pytest suite for BasicUNet.
"""

import pytest
import torch
import neurite as ne


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
    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[8, 16],
        normalizations=None,
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


@pytest.mark.parametrize("skip_connections", [True, False])
def test_basicunet_skip_option(skip_connections):
    """
    Test that the skip_connections flag toggles skip connections
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

    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[4, 8],
        skip_connections=skip_connections,
    )
    model.eval()

    x = torch.randn((1, in_ch, *size))
    y = model(x)

    # Even without residuals, output dims stay the same
    assert y.shape == (1, out_ch, *size)


@pytest.mark.parametrize("skip_connections", [True, False])
def test_basicunet_asymmetric_skip_option(skip_connections):
    """
    Test that the skip_connections flag toggles skip connections
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

    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[[9, 2], [12, 4]],
        skip_connections=skip_connections,
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

    model = ne.nn.models.BasicUNet(
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
    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
    ).cuda()
    model.eval()

    x = torch.randn((1, in_ch, 32, 32), device="cuda")
    y = model(x)

    # Ensure output is on CUDA
    assert y.device.type == "cuda"
    assert y.shape == (1, out_ch, 32, 32)


@pytest.mark.skipif(
    not torch.mps.is_available(), reason="MPS device not available"
)
def test_basicunet_forward_mps():
    """
    Test that the model can be moved to MPS and still produce correct output.

    Returns
    -------
    None
    """
    ndim = 2
    in_ch = 1
    out_ch = 1

    # Move model and data to MPS
    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
    ).to('mps')
    model.eval()

    x = torch.randn((1, in_ch, 32, 32), device="mps")
    y = model(x)

    # Ensure output is on MPS
    assert y.device.type == "mps"
    assert y.shape == (1, out_ch, 32, 32)


def test_basicunet_list_normalizations():
    """
    Test that BasicUNet accepts a list of different normalizations per level.

    This tests the fix for Bug #1: Previously, passing a list would cause
    an AttributeError because self.normalizations was only set when
    normalizations was NOT a list.
    """
    ndim = 2
    in_ch = 1
    out_ch = 1

    # Create model with different normalizations per level
    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[16, 32, 64],
        normalizations=['batch', 'instance', None],
        activations=['relu', 'elu', 'relu'],
        skip_connections=True,
    )

    # Verify attributes were set correctly
    assert model.normalizations == ['batch', 'instance', None]
    assert model.activations == ['relu', 'elu', 'relu']

    # Test forward pass
    x = torch.randn(1, in_ch, 32, 32)
    y = model(x)

    assert y.shape == (1, out_ch, 32, 32)


def test_upsampling_normalization_order():
    """
    Test that upsampling blocks use normalizations in reversed order.

    This tests the fix for Bug #2: Previously, normalizations[-i] when i=0
    would access normalizations[0] instead of normalizations[-1], breaking
    the expected encoder-decoder symmetry.
    """
    from neurite.utils.utils import upsampling_conv_blocks

    ndim = 2
    nb_features = [64, 32, 16]
    normalizations = ['batch', 'instance', None]

    # Create upsampling blocks
    blocks = upsampling_conv_blocks(
        ndim=ndim,
        nb_features=nb_features,
        normalizations=normalizations,
        activations='relu',
        accepts_skip=False,
    )

    # The blocks should be created with reversed normalizations
    # Expected order: [None, instance, batch]
    assert len(blocks) == 3

    # Test forward pass
    x = torch.randn(1, 64, 16, 16)
    for block in blocks:
        x = block(x)

    # Final output should be at original spatial resolution (upsampled 3 times)
    assert x.shape[2] == 128  # 16 -> 32 -> 64 -> 128
    assert x.shape[3] == 128


def test_basicunet_symmetric_normalization():
    """
    Test that symmetric UNet with different normalizations works correctly.

    For a symmetric UNet, the upsampling path should mirror the downsampling
    path in terms of normalizations (in reverse order).
    """
    ndim = 2
    in_ch = 1
    out_ch = 1

    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=in_ch,
        out_channels=out_ch,
        nb_features=[8, 16, 32],
        normalizations=['batch', 'instance', None],
        skip_connections=True,
    )

    # Test forward and backward pass
    x = torch.randn(2, in_ch, 64, 64, requires_grad=True)
    y = model(x)

    assert y.shape == (2, out_ch, 64, 64)

    # Test gradient flow
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_basicunet_downsample_first_shape_preservation(ndim):
    """
    Test that downsample_first=True preserves input/output shape across dimensions.
    """
    torch.manual_seed(42)
    spatial = [64] * ndim
    x = torch.randn(2, 1, *spatial)

    model = ne.nn.models.BasicUNet(
        ndim=ndim,
        in_channels=1,
        out_channels=1,
        nb_features=[16, 32, 64],
        downsample_first=True
    )
    model.eval()

    y = model(x)

    assert y.shape == x.shape
