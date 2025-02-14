"""
Losses for the neurite project.

Citation
--------
If you use this code, please cite the following, and read function docs for further info/citations
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018. https://arxiv.org/abs/1903.03148

License
-------
Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""
__all__ = [
    "Dice",
    "SoftDice",
    "HardDice",
    "CategoricalCrossentropy",
    "MeanSquaredErrorProb",
]

import torch
from torch import nn


class Dice(nn.Module):
    """
    Compute the (hard or soft) Dice Coefficient between two tensors.
    """
    def __init__(self):
        """
        Initialize the `Dice` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Dice` module.
        """
        raise NotImplementedError("The `Dice` module isn't ready yet :(")


class SoftDice(nn.Module):
    """
    Compute the Soft Dice Coefficient between two tensors.
    """
    def __init__(self):
        """
        Initialize the `SoftDice` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `SoftDice` module.
        """
        raise NotImplementedError("The `SoftDice` module isn't ready yet :(")


class HardDice(nn.Module):
    """
    Compute the Hard Dice Coefficient between two tensors.
    """
    def __init__(self):
        """
        Initialize the `HardDice` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `HardDice` module.
        """
        raise NotImplementedError("The `HardDice` module isn't ready yet :(")


class CategoricalCrossentropy(nn.Module):
    """
    Compute the Categorical Crossentropy between two tensors.
    """
    def __init__(self):
        """
        Initialize the `CategoricalCrossentropy` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `CategoricalCrossentropy` module.
        """
        raise NotImplementedError("The `CategoricalCrossentropy` module isn't ready yet :(")


class MeanSquaredErrorProb(nn.Module):
    """
    Compute the Mean Squared Error between two tensors.
    """
    def __init__(self):
        """
        Initialize the `CLASSNAME` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `CLASSNAME` module.
        """
        raise NotImplementedError("The `CLASSNAME` module isn't ready yet :(")
