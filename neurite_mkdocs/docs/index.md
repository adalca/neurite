site_name: neurite
# Welcome to neurite!

Neurite is an open‐source, PyTorch‐based library designed to accelerate deep learning research and
development. It provides a rich set of modular building blocks—from advanced [cross convolution]
[neurite.pytorch.modules.CrossConvBlock] layers to simple segmentation loss functions like [Dice]
[neurite.pytorch.losses.Dice]—empowering you to rapidly prototype and deploy state‐of‐the‐art
neural network architectures.

# Resources
* [GitHub repository for `neurite`](https://github.com/adalca/neurite)

# Related Publications
* [_Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation_](https://arxiv.org/abs/1903.03148)
* [_UniverSeg: Universal Medical Image Segmentation_](https://arxiv.org/abs/2304.06131)
* [_Tyche: Stochastic In-Context Learning for Medical Image Segmentation_](https://doi.org/10.48550/arXiv.2401.13650)

``` mermaid
graph LR
  A[Start] --> B{Error?};
  B -->|Yes| C[Hmm...];
  C --> D[Debug];
  D --> B;
  B ---->|No| E[Yay!];
```

??? reference "Related Publications"

    * [_Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation_](https://arxiv.org/abs/1903.03148)
    * [_UniverSeg: Universal Medical Image Segmentation_](https://arxiv.org/abs/2304.06131)
    * [_Tyche: Stochastic In-Context Learning for Medical Image Segmentation_](https://doi.org/10.48550/arXiv.2401.13650)
