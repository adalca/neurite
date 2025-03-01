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
import neurite as ne


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
    Soft dice loss for single and multi-class segmentation with adjustable slope.

    Computes the soft dice loss for each class separately and returns the mean loss across classes
    and batches. An adjustable slope parameter scales the logits before applying the sigmoid
    function, effectively controlling the sharpness of the prediction probabilities.

    Examples
    --------
    >>> loss_fn = SoftDiceLoss(slope=2.0)
    >>> logits = ne.samplers.Normal(0, 1)(shape)
    >>> targets = ne.samplers.RandInt(0, 1)(shape)
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, slope: float = 1.0, smooth: float = 1e-6) -> None:
        """
        Instantiate `SoftDice`

        Parameters
        ----------
        slope : float, optional
            Scaling factor for the slope of the sigmoid function. A higher value makes the sigmoid
            function steeper, by default 1.0.
        smooth : float, optional
            Smoothing constant to avoid division by zero, by default 1e-6.
        """

        super().__init__()
        self.slope = slope
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft dice loss for one or multiple classes.

        Parameters
        ----------
        logits : torch.Tensor
            Raw output logits/score from the network with shape (batch_size, channels, ...).
            Expected to be unbounded.
        targets : torch.Tensor
            Ground truth labels with shape (batch_size, channels, ...). Must be binary or one-hot
            encoded for each class.

        Returns
        -------
        torch.Tensor
            The mean soft dice loss computed across all classes and batches.
        """

        # Obtain probabilities by passing logits through custom-slope sigmoid
        probs = ne.logistic(logits, self.slope)

        # Flatten spatial dimensions while preserving batch and channel dims
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Per-class intersection
        intersection = (probs * targets).sum(dim=2)

        # Per-class union
        union = probs.sum(dim=2) + targets.sum(dim=2)

        # Compute the dice score with intersection, smooth, & union
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average loss over classes and batches.
        # TODO: Optionally make this mean, max, min (but only useful for multiclass so maybe not)
        loss = 1 - dice_score.mean()

        return loss


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
