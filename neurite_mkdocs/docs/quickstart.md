# Installation
We strongly recommend creating a dedicated environment for Neurite using Mamba with Python 3.9.

### Step 1: Install mamba
If you haven't installed Mamba yet, follow the detailed instructions in the official [Miniforge repo](https://github.com/conda-forge/miniforge).

### Step 2: Create and activate the environment
Run the following commands to set up and activate your `neurite` environment with Python 3.9 and
GPU-enabled PyTorch.

```bash
mamba create -n neurite python~=3.9 pytorch-cuda torchvision torchaudio cudatoolkit ipykernel ipython -c pytorch -c nvidia
mamba activate neurite
```

### Step 3: Install neurite
Install neurite from PyPi with:

```bash
pip install neurite
```

# Usage

Here's a quick demonstration using neurite to make a basic [UNet][neurite.pytorch.models.BasicUNet],
sample some random segmentation data (with neurite's [random samplers][neurite.pytorch.samplers]),
and use the [dice score][neurite.pytorch.losses.Dice] to assess performance.

=== "Simple Demo"    
    Here's a quick demonstration using neurite to make a basic [unet][neurite.pytorch.models.BasicUNet],
    sample some random segmentation data (with neurite's [random samplers][neurite.pytorch.samplers]),
    and use the [dice score][neurite.pytorch.losses.Dice] to assess performance. The three arguments
    passed to `BasicUNet` are the minimal necessary arguments.

    ```python
    import neurite as ne

    # Instantiate a basic unet
    model = ne.models.BasicUNet(
        ndim=3,
        in_channels=2,
        out_channels=5,
    )

    # Make some random input data for e.g. supervised segmentation
    x_data = ne.Normal(0, 1)((batch_size, 2, 64, 64, 64))
    y_data = ne.RandInt(0, 1)((batch_size, 5, 64, 64, 64))

    # Make a prediction
    prediction = model(x_data)

    # Instantiate a performance metric
    dice_module = ne.losses.Dice()

    # Compute the performance metric
    dice_module(prediction, y_data)
    ```

=== "Custom Activations"
    All you really need to do for custom activations is add an `activation` parameter to `BasicUNet`.
    This can either be a list of [torch activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity),
    or a string (options [here][neurite.pytorch.modules.Activation]). If you pass a list, it there must be the same number of elements as there are "a"'s in the `order` parameter.

    ```python
    import neurite as ne

    # Instantiate a basic unet
    model = ne.models.BasicUNet(
        ndim=3,
        in_channels=2,
        out_channels=5,
        order='cnacna
        activation=[nn.PReLU, 'relu']
    )

    # Make some random input data for e.g. supervised segmentation
    x_data = ne.Normal(0, 1)((batch_size, 2, 64, 64, 64))
    y_data = ne.RandInt(0, 1)((batch_size, 5, 64, 64, 64))

    # Make a prediction
    prediction = model(x_data)

    # Instantiate a performance metric
    dice_module = ne.losses.Dice()

    # Compute the performance metric
    dice_module(prediction, y_data)
    ```

=== "Custom Features"
    To customize the number of features per layer, just pass a tuple or list into the argument `nb_features`!
    This will make you a unet with `len(nb_features)` levels, with each level containing the number
    of feature extractors/kernels specified by its index in the list. For example, this UNet has 3
    layers, and each layer has 32 feature extractors.

    ```python
    import neurite as ne

    # Instantiate a basic unet
    model = ne.models.BasicUNet(
        ndim=3,
        in_channels=2,
        out_channels=5,
        nb_features=(32, 32, 32),
    )

    # Make some random input data for e.g. supervised segmentation
    x_data = ne.Normal(0, 1)((batch_size, 2, 64, 64, 64))
    y_data = ne.RandInt(0, 1)((batch_size, 5, 64, 64, 64))

    # Make a prediction
    prediction = model(x_data)

    # Instantiate a performance metric
    dice_module = ne.losses.Dice()

    # Compute the performance metric
    dice_module(prediction, y_data)
    ```

---
# Next steps

For more advanced usage, customization options, and flexibility, refer to the jupyter notebooks in
the documentation of this project, or, just click on the API reference in the sidebar of this page!
