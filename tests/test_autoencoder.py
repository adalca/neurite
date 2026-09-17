"""Regression tests for BasicAutoencoder construction and configuration."""

import pytest
import torch

import neurite as ne

AUTOENCODER_CONFIGS = [
    (None, None, torch.nn.Identity),
    ("batch", "elu", torch.nn.ELU),
    (torch.nn.BatchNorm1d, torch.nn.LeakyReLU, torch.nn.LeakyReLU),
    (["batch", "instance"], ["relu", "elu"], torch.nn.ELU),
    (("batch", "instance"), ("relu", "elu"), torch.nn.ELU),
]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("nb_features", [(4,), (4, 8, 6), (16, 16, 16, 16, 16)])
def test_basic_autoencoder_forward_backward(device, ndim, nb_features):
    """Check shapes and gradients for single-level, unequal, and default feature counts."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    model = ne.nn.models.BasicAutoencoder(
        ndim=ndim,
        in_channels=2,
        latent_features=3,
        out_channels=1,
        nb_features=nb_features,
    )
    model = model.to(device)
    spatial_shape = (2 ** len(nb_features),) * ndim
    input_tensor = torch.randn(2, 2, *spatial_shape, device=device, requires_grad=True)

    output = model(input_tensor)
    assert output.shape == (2, 1, *spatial_shape)
    assert output.device == input_tensor.device
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("normalizations, activations, latent_activation", AUTOENCODER_CONFIGS)
def test_basic_autoencoder_configuration(normalizations, activations, latent_activation):
    """Accept scalar and per-level options and honor the deepest latent activation."""
    model = ne.nn.models.BasicAutoencoder(
        ndim=1,
        in_channels=1,
        latent_features=3,
        out_channels=1,
        nb_features=(4, 6),
        normalizations=normalizations,
        activations=activations,
        order="cna",
    )
    input_tensor = torch.randn(2, 1, 16, requires_grad=True)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert torch.isfinite(input_tensor.grad).all()

    # A negative convolution output distinguishes ELU, identity, and LeakyReLU from ReLU.
    latent_layer = model.downsampling_conv_blocks[-1]
    with torch.no_grad():
        latent_layer.conv0.weight.zero_()
        latent_layer.conv0.bias.fill_(-1)
        latent_input = torch.zeros(2, 6, 4)
        actual = latent_layer(latent_input)
        preactivation = torch.full_like(actual, -1)
        expected = latent_activation()(preactivation)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("padding_mode", ["replicate", "reflect"])
def test_basic_autoencoder_padding(padding_mode):
    """Nonzero padding modes preserve a constant input through averaging convolutions."""
    model = ne.nn.models.BasicAutoencoder(
        ndim=1,
        in_channels=1,
        latent_features=1,
        out_channels=1,
        nb_features=(1, 1),
        activations=None,
        padding_mode=padding_mode,
    )

    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv1d):
                layer.weight.fill_(1 / layer.kernel_size[0])
                layer.bias.zero_()
        input_tensor = torch.ones(1, 1, 16)
        output = model(input_tensor)
    torch.testing.assert_close(output, input_tensor)
