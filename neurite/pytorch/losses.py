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
    "CategoricalCrossentropy",
    "MeanSquaredErrorProb",
]

import torch
from torch import nn
import neurite.pytorch as ne


class Dice(nn.Module):
    """
    Compute the Dice score between two segmentation tensors (e.g. ground truth, predictions, etc...)

    Examples
    --------
    # Example 1: Computing the hard dice score with binary seg maps
    >>> # Instantiate the dice score module
    >>> dice_module = ne.losses.Dice()
    >>> # Randomly sample binary tensors with 3 batches and 4 channels
    >>> seg1 = ne.samplers.RandInt(0, 1)((3, 4, 128, 128))
    >>> seg2 = ne.samplers.RandInt(0, 1)((3, 4, 128, 128))
    >>> # Compute the dice score and return
    >>> dice_module(seg1, seg2)
    tensor([[0.5003]])
    
    # Example 2: Computing the soft dice score with continuious seg maps and no reduction
    >>> dice_module = Dice(reduction=None)
    >>> # Randomly sample continuious "logits"
    >>> seg1 = ne.samplers.Normal(0, 1)((3, 4, 128, 128))
    >>> seg2 = ne.samplers.Normal(0, 1)((3, 4, 128, 128))
    >>> # Activation functions
    >>> seg1 = ne.pytorch.utils.logistic(seg1)
    >>> seg2 = ne.pytorch.utils.logistic(seg2)
    >>> # Compute the dice score and return
    >>> dice_module(seg1, seg2)
    tensor([[0.4982, 0.5022, 0.4984, 0.5024],
            [0.5016, 0.5035, 0.5021, 0.5001],
            [0.5001, 0.4998, 0.4990, 0.4996]])
    """

    def __init__(
        self,
        smooth_numerator: float = 1e-12,
        smooth_denominator: float = 1e-12,
        reduction: str = 'mean',
        reduction_dim: int = (0, 1),
        keepdims: bool = True,
    ) -> None:

        """
        Initialize the `Dice` module.

        Parameters
        ----------
        smooth_numerator : float, optional
            Smoothing constant added to the numerator.
        smooth_denominator : float, optional
            Smoothing constant added to the denominator.
        reduction : str, optional
            The type of reduction to apply. Supported values for multidimensional reductions are:
            'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
            reductions: 'argmin', 'argmax', and all multidimensionals. Default is 'mean'.
        dim : int or tuple of ints, optional
            Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
            tuple of dimensions; for single-dimension reductions, pass an integer. Default is (0, 1)
        keepdims : bool, optional
            Whether to retain reduced dimensions as a singleton. Default is False.
        """
        super().__init__()

        # Store attributes
        self.smooth_numerator = smooth_numerator
        self.smooth_denominator = smooth_denominator
        self.reduction = reduction
        self.reduction_dim = reduction_dim
        self.keepdims = keepdims

    def forward(self, seg1: torch.Tensor, seg2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice coefficient between two segmentation tensors.

        Parameters
        ----------
        seg1 : torch.Tensor
            First segmentation tensor of shape (B, C, *spatial_dims).
        seg2 : torch.Tensor
            Second segmentation tensor with the same shape as `seg1`

        Returns
        -------
        torch.Tensor
            The computed Dice coefficient, potentially reduced according to the arguments passed at
            point of object instantiation.
        """

        # Compute the dice score
        dice_score = ne.utils.dice(
            seg1=seg1,
            seg2=seg2,
            smooth_numerator=self.smooth_numerator,
            smooth_denominator=self.smooth_denominator
        )

        # Reduce the score if necessary and return
        if self.reduction is None:
            return dice_score
        else:
            return ne.pytorch.utils.reduce_tensor(
                tensor=dice_score,
                reduction=self.reduction,
                dim=self.reduction_dim,
                keepdims=self.keepdims
            )


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
