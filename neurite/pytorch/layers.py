"""
Layers for the neurite project, written in PyTorch.

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
    "RescaleValues",
    "Resize",
    "SoftQuantize",
    "MSE",
    "GaussianBlur",
    "Resample",
    "RandomCrop",
    "RandomClip",
    "RandomGamma",
    "RandomIntensityLookup",
    "RandomClearLabel",
    "SampleImageFromLabels",
]

from typing import Optional, Union, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F

import neurite.pytorch as ne


class RescaleValues(nn.Module):
    """
    Scale each element of the input tensor by a multiplicative factor.
    """

    def __init__(self, scale_factor: Union[float, int, ne.samplers.Sampler]):
        """
        Initialize the `RescaleValues` module.

        Parameters
        ----------
        scale_factor : float, int, or Sampler
            Factor (or sampler) by which to rescale the values of the input tensor.
        """
        super().__init__()

        # Declare the scale factor as a sampled quantity
        self.scale_factor = ne.samplers.make_sampler(ne.samplers.Fixed, scale_factor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RescaleValues` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor to be resacled.

        Returns
        -------
        torch.Tensor
            Rescaled tensor.
        """
        return input_tensor * self.scale_factor()


class Resize(nn.Module):
    """
    Resize the input tensor.
    """
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
    ):
        """
        Initialize the `Resize` module.

        Parameters
        ----------
        size : int or Tuple[int, int], optional
            The desired output size. If None, uses `scale_factor`.
        scale_factor : float or Tuple[float, float], optional
            Scaling factor for resizing. If None, uses `size`.
        mode : str, default="nearest"
            Interpolation mode (e.g., "nearest", "bilinear").
        align_corners : bool, optional
            Alignment for "linear", "bilinear", or "trilinear" modes.
        recompute_scale_factor : bool, optional
            If True, recomputes the scale factor for interpolation.
        antialias : bool, default=False
            Applies anti-aliasing if `scale_factor` < 1.0.

        Examples
        --------
        >>> # Define a tensor we'll use for resizing examples
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)

        ### Resizing with fixed `scale_factor`
        >>> resize_module = Resize(scale_factor=2)
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor.shape)
        torch.Size([1, 1, 64, 64, 64])

        ### Resizing with a sampled `scale_factor`
        >>> resize_module = Resize(scale_factor=Uniform(0.5, 4))
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 74, 74, 74])

        ### Resizing to a specific size
        >>> resize_module = Resize(size=(96, 96, 96))
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 96, 96, 96])

        Notes
        -----
        - This class assumes the input tensor has batch and channel dimensions.
        - It is not possible to define `size` and `scale_factor` simultaneously. Only one can be
        defined for a given instatntiation of `Resize`.
        - When defining `size` do not include batch or channel dimensions, only spatial dims.
        """
        super().__init__()

        # Either scale factor or size must be defined. If neither is, make scale factor fixed @ 1.
        if size is None and scale_factor is None:
            scale_factor = 1

        elif scale_factor is not None:
            # Make a fixed if passed a single number. Makes sampler if passed sampler.
            scale_factor = ne.samplers.make_sampler(ne.samplers.Fixed, scale_factor)

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Resize` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be resized. Must have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            The resized tensor.
        """
        resized_tensor = F.interpolate(
            input=input_tensor,
            size=self.size,
            scale_factor=self.scale_factor(),
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )

        return resized_tensor


class SoftQuantize(nn.Module):
    """
    Map continuous values to discrete bins.

    This module maps continuous values to discrete bins while retaining some smoothness/continuous
    which is parametrized by a softening parameter. It is especially useful in the context of
    machine learning, where it is desirable to have a differentiable version of a quantized
    quantity, allowing for backprop. Hard quantization is non-differentiable and creates gradients
    of zero, making gradient-based optimization impossible.
    """

    def __init__(
        self,
        nb_bins: Union[int, ne.samplers.Sampler] = 16,
        softness: Union[float, int, ne.samplers.Sampler] = 1.0,
        min_clip: Union[float, int, ne.samplers.Sampler] = -float('inf'),
        max_clip: Union[float, int, ne.samplers.Sampler] = float('inf'),
        return_log: bool = False,
    ):
        """
        Initialize the `SoftQuantize` module.

        Parameters
        ----------
        nb_bins : int or Sampler, optional
            The number of discrete bins to softly quantize the input values into. By default 16
        softness : float, int, or Sampler, optional
            The softness factor for quantization. A higher value gives smoother quantization.
            By default 1.0
        min_clip : float, int, or Sampler, optional
            Clip data lower than this value before calculating bin centers. By default -float('inf')
        max_clip : float, int, or Sampler, optional
            Clip data higher than this value before calculating bin centers. By default float('inf')
        return_log : bool, optional
            Optionally return the log of the softly quantized tensor. By default False

        Examples
        --------
        >>> # Make a random 3D tensor with zero mean and unit variance.
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> # Initialize the SoftQuantize instance.
        >>> soft_quantizer = SoftQuantize(nb_bins=4, softness=0.5)
        >>> # Apply the SoftQuantize instance to the input tensor
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> # Visualize the softly quantized tensor.
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])

        ### Softly quantizing with randomly sampled `nb_bins` and `softness` parameters
        >>> # Define `nb_bins` to sample a uniform int distribution, and `softness` a float dist
        >>> soft_quantizer = SoftQuantize(nb_bins=RandInt(3, 32), softness=Uniform(0.001, 10))
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])
        """
        super().__init__()
        self.nb_bins = nb_bins
        self.softness = softness
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.return_log = return_log

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `SoftQuantize` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to softly quantize.

        Returns
        -------
        torch.Tensor
            Softly quantized tensor with the same dimensions as `input_tensor`.
        """

        return ne.utils.soft_quantize(
            input_tensor=input_tensor,
            nb_bins=self.nb_bins,
            softness=self.softness,
            min_clip=self.min_clip,
            max_clip=self.max_clip,
            return_log=self.return_log
        )


class MSE(nn.Module):
    """
    Calculate the mean squared error.
    """

    def __init__(self):
        """
        Initialize the `MSE` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `MSE` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor representing the model's prediction(s).
        target_tensor : torch.Tensor
            The target or ground truth values.

        Returns
        -------
        torch.Tensor
            The mean squared error between `input_tensor` and `target_tensor`.
        """

        return ne.utils.mse(input_tensor=input_tensor, target_tensor=target_tensor)


class GaussianBlur(nn.Module):
    """
    Apply a {1D, 2D, 3D} gaussian blur to the input tensor by convolving it with a Gaussian kernel.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1,
    ):
        """
        Initialize the `GaussianBlur` module.

        Parameters
        ----------
        kernel_size : Sampler or int, optional
            Size of the Gaussian kernel, default is 3.
        sigma : float, int, or Sampler, optional
            Standard deviation of the Gaussian kernel, default is 1.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `GaussianBlur` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor, assumed to be 1D, 2D, or 3D.

        Returns
        -------
        torch.Tensor
            The smoothed tensor.
        """

        return ne.utils.gaussian_smoothing(
            input_tensor=input_tensor,
            kernel_size=self.kernel_size,
            sigma=self.sigma
        )


class Resample(nn.Module):
    """
    Spatially resample {subsample, resample} the input tensor.

    This module resamples the input tensor by a factor of `stride` along the
    specified dimension by interleaving dropouts along it (keeping every `stride`th element).
    Optionally upsample the tensor after downsampling it to restore it to its original dimensions.
    """

    def __init__(
        self,
        upsample: bool = True,
        stride: Union[int, Tuple[int, int]] = 2,
        forbidden_dims: Tuple[int, int] = (1, 0),
        p: float = 0.5,
        max_concurrent_subsamplings: int = None,
        mode: str = 'nearest',
    ):
        """
        Initialize the `Resample` module.

        Parameters
        ----------
        upsample : bool, optional
            optionally upsample the subsampled tensor to the original dimensions of `input_tensor`.
            By default, True.
        stride : Sampler or int or tuple, optional
            The stride value to use when subsampling a given dimension. Can be an integer or
            a tuple corresponding to the range of strides to sample. By default, 2.
                - A stride of 1 does not result in any subsampling.
                - A stride of 2 will reduce the elements of the selected dimension by 1/2.
        forbidden_dims : list, optional
            A list of dimensions that should not be subsampled. If None, no dimensions
            are forbidden from subsampling. Default is (0, 1) to ignore batch and channel
            dimensions.
        p : float, optional
            The probability of selecting each dimension for subsampling. This probability
            is applied as an independent Bernoulli trial for each dimension. By default, 0.5.
        max_concurrent_subsamplings : int, optional
            The maximum number of dimensions that can be subsampled simultaneously. If
            None, the number of concurrent subsamplings is set to the number of dimensions
            in `input_tensor`. Default is None.
        mode : str, optional
            The interpolation mode to use for upsampling. By default None. Options (WRT spatial
            dimensions) include:
                - 'nearest' (default)
                - 'linear' (1D-only)
                - 'bilinear' (2D-only)
                - 'bicubic' (2D-only)
                - 'trilinear' (3D-only)
                - 'area'

        Examples
        --------
        ### Custom stride and only subsampling
        >>> # Initialize a random 3D tensor with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128, 128)
        >>> # Resample the tensor with random strides on the (inclusive) interval (2, 5)
        >>> resampled_tensor = Resample(upsample=False, stride=(2, 5))(input_tensor)
        >>> # Spatial dimensions be different
        >>> print(resampled_tensor.shape)
        torch.Size([1, 1, 64, 128, 32])

        ### Custom stride with upsampling and trilinear interpolation
        >>> # Initialize a random 3D tensor with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128, 128)
        >>> # Resample the tensor with a stride upper bound of 6, trilinear interpolation, and with
        a final upsampling operation after downsampling.
        >>> resampled_tensor = Resample(stride=6, mode='trilinear')(input_tensor)
        >>> # Spatial dims should be the same (because the last operation is upsampling to original)
        >>> print(input_tensor.shape)
        torch.Size([1, 1, 128, 128, 128])
        """
        super().__init__()
        self.upsample = upsample
        self.stride = stride
        self.forbidden_dims = forbidden_dims
        self.p = p
        self.max_concurrent_subsamplings = max_concurrent_subsamplings
        self.mode = mode

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Resample` module.
        """
        # Store original spatial shape to restore dimensions in upsampling
        resampled_tensor = input_tensor
        original_spatial_shape = input_tensor.shape[2:]

        # Start by subsampling the input tensor
        resampled_tensor = ne.utils.subsample_tensor_random_dims(
            input_tensor=resampled_tensor,
            stride=self.theta.get('stride'),
            forbidden_dims=self.theta.get('forbidden_dims'),
            p=self.theta.get('p'),
            max_concurrent_subsamplings=self.theta.get('max_concurrent_subsamplings')
        )

        # Optionally upsample the resuling subsampled tensor
        if self.upsample:
            # Apply upsampling
            resampled_tensor = ne.utils.upsample_tensor(
                resampled_tensor,
                original_spatial_shape
            )

        return resampled_tensor


class RandomCrop(nn.Module):
    """
    Randomly crop the input tensor to a particular field of view.

    This module randomly selects a subset of the allowed dimensions (excluding `forbidden_dims`)
    and crops each independently by a proportion that is randomly drawn from a distribution. The
    proportion to crop can either be fixed or sampled from a specified distribution. Each allowed
    dimension has a probability `prob` of being cropped based on the results of independent
    Bernoulli trials.
    """

    def __init__(
        self,
        crop_proportion: Union[ne.samplers.Sampler, float] = 0.5,
        prob: Union[ne.samplers.Sampler, float] = 1,
        forbidden_dims: ne.samplers.Union[Tuple, List] = (0, 1),
        seed: Union[int, ne.samplers.Sampler] = None,
    ):
        """
        Initialize the `RandomCrop` module.

        Parameters
        ----------
        crop_proportion : Union[Sampler, float], optional
            The proportion that is randomly cropped from any allowed dimension. By default 0.5
            - If a `float` is provided, it represents the maximum proportion (0 to 1) to crop,
            sampled from independent uniform distributinos for each allowed dimension. A value of
            `0.5` means up to 50% of each dimension can be cropped.
            - If a `Sampler` is provided, cropped proportions are dynamically sampled based on the
            specified distribution
        prob : Union[Sampler, float], optional
            The probability of cropping each allowed dimension. By default 1.0
            - If a `float` is provided, it's used as a fixed probability for all eligible
            dimensions.
            - If a `Sampler` is provided, probabilities are dynamically generated for each
            dimension.
        forbidden_dims : Union[Tuple[int, ...], List[int]], optional
            Dimensions that should never be cropped. By defult `(0, 1)` (batch and channel
            dimensions)
        seed : Union[int, Sampler], optional
            A random seed or sampler to control the randomness of cropping operations. If provided,
            it ensures reproducibility of the cropping. Defaults to `None`.
        """
        super().__init__()
        self.crop_proportion = crop_proportion
        self.prob = prob
        self.forbidden_dims = forbidden_dims
        self.seed = seed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomCrop` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor to be randomly cropped. It is assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            The tensor that has been randomly cropped.
        """

        return ne.utils.augment.random_crop(
            input_tensor=input_tensor,
            crop_proportion=self.crop_proportion,
            prob=self.prob,
            forbidden_dims=self.forbidden_dims,
            seed=self.seed
        )


class RandomClip(nn.Module):
    """
    Randomly clip the intensities of the input tensor.
    """

    def __init__(
        self,
        clip_min: Union[float, int, ne.samplers.Sampler] = 0,
        clip_max: Union[float, int, ne.samplers.Sampler] = 1,
        clip_prob: Union[float, int, ne.samplers.Sampler] = 0.5,
        seed: Union[int, ne.samplers.Sampler] = None,
    ):
        """
        Initialize `RandomClip` with specified clipping bounds and sampling probability.

        Parameters
        ----------
        clip_min : Union[float, int, Sampler], optional
            The lower bound for clipping. Elements less than `clip_min` are set to `clip_min`.
            Defaults to 0.
        clip_max : Union[float, int, Sampler], optional
            The upper bound for clipping. Elements greater than `clip_max` are set to `clip_max`.
            Defaults to 1.
        clip_prob : Union[float, int, Sampler], optional
            Probability of applying this operation. Defaults to 0.5.
        seed : Union[int, Sampler], optional
            Seed for random number generation to ensure reproducibility. Defaults to None.

        Examples
        --------
        ### Initialize the `RandomClip` module and apply it to a tensor:
        >>> random_clip = RandomClip(clip_min=0.1, clip_max=0.9, clip_prob=0.5)
        >>> input_tensor = torch.randn(3, 3)
        >>> output_tensor = random_clip(input_tensor)
        >>> print(output_tensor)

        ### Use a sampler for dynamic clipping bounds:
        >>> from my_samplers import UniformSampler
        >>> random_clip = RandomClip(
                clip_min=UniformSampler(0, 0.5),
                clip_max=UniformSampler(0.5, 1.0)
            )
        >>> output_tensor = random_clip(input_tensor)
        >>> print(output_tensor)
        """
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_prob = clip_prob
        self.seed = seed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomClip` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be clipped.

        Returns
        -------
        torch.Tensor
            The clipped tensor (if Bernoulli trial defined by parameter `clip_prob` is successful).
        """

        return ne.utils.augment.random_clip(
            input_tensor=input_tensor,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            clip_prob=self.clip_prob,
            seed=self.seed
        )


class RandomGamma(nn.Module):
    """
    Random nonlinear gamma scaling operation.

    The gamma scaling operation adjusts the contrast of the input tensor by applying a non-linear
    operation. Specifically, each element in the tensor is raised to the power of `gamma`. This can
    enhance or diminish the contrast of the input data, making it a valuable augmentation tool for
    various deep learning tasks.
    """

    def __init__(
        self,
        gamma: Union[float, int, ne.samplers.Sampler] = 1.0,
        prob: Union[float, int, ne.samplers.Sampler] = 1.0,
        seed: Union[int, ne.samplers.Sampler] = None,
    ):
        """
        Initialize the `RandomGamma` module.

        Parameters
        ----------
        gamma : Union[float, int, Sampler], optional
            The gamma value to apply for the scaling operation.
            - If a `float` is provided, it represents a fixed gamma value.
            - If a `Sampler` is provided, the gamma value is dynamically sampled from the specified
            distribution.
            By default `1.0`, which leaves the tensor unchanged.
        prob : Union[float, int, Sampler], optional
            The probability of applying the gamma operation.
            - If a `float` is provided, it's used as a fixed probability for the operation.
            - If a `Sampler` is provided, probabilities are dynamically generated for each
            invocation.
            Default is `1.0` (always apply).
        seed : Union[int, Sampler], optional
            A random seed or sampler to control the randomness of the gamma scaling operation. If
            provided, it ensures reproducibility of the operation. Defaults to `None`.

        Examples
        --------
        ### Fixed gamma scaling operation
        >>> gamma_module = RandomGamma(gamma=2.0, prob=1.0)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor = gamma_module(tensor)
        >>> print(gamma_tensor)
        tensor([0.0625, 0.2500, 0.5625])

        ### Randomized gamma scaling with a range of gamma values
        >>> gamma_sampler = Uniform(0.5, 1.5)
        >>> gamma_module = RandomGamma(gamma=gamma_sampler, prob=0.8)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor = gamma_module(tensor)
        >>> print(gamma_tensor)
        tensor([0.1768, 0.5000, 0.8367])

        ### Applying gamma operation with reproducibility
        >>> gamma_module1 = RandomGamma(gamma=2.0, prob=1.0, seed=42)
        >>> gamma_module2 = RandomGamma(gamma=2.0, prob=1.0, seed=42)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor1 = gamma_module1(tensor)
        >>> gamma_tensor2 = gamma_module2(tensor)
        >>> print(torch.equal(gamma_tensor1, gamma_tensor2))
        True
        """
        super().__init__()
        self.gamma = gamma
        self.prob = prob
        self.seed = seed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomGamma` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to which the gamma operation will be applied. It is assumed to
            have a range suitable for gamma correction (typically normalized between 0 and 1).

        Returns
        -------
        torch.Tensor
            The tensor after applying the gamma scaling operation. If operation is not applied
            (based on `prob`), the original `input_tensor` is returned unchanged.
        """

        return ne.utils.augment.random_gamma(
            input_tensor=input_tensor,
            gamma=self.gamma,
            prob=self.prob,
            seed=self.seed
        )


class RandomIntensityLookup(nn.Module):
    """
    Randomly augment the contrast of a single-channel tensor.

    Compute a smoothly varying lookup table to map the original single-channel tensor (usually a
    greyscale image) to a tensor with a new contrast.
    """

    def __init__(self):
        """
        Initialize the `RandomIntensityLookup` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomIntensityLookup` module.
        """
        raise NotImplementedError("The `RandomIntensityLookup` module isn't ready yet :(")


class RandomClearLabel(nn.Module):
    """
    Randomly clear/erase regions from an image corresponding to randomly
    selected entities/continuious regions in a label map.

    Identifies unique labels within the `label_tensor` and, based on a specified
    probability, designates regions of the `input_tensor` to be cleared (set to zero) corresponding
    to randomly selected labels. This can be used for tasks such as data augmentation, where certain
    labels are randomly omitted to simulate occlusions or missing annotations.

    Examples
    --------
    ### Clearing labels with a fixed probability
    >>> clear_label = RandomClearLabel(prob=0.75)
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor = clear_label(input_tensor, label_tensor)
    >>> print(cleared_tensor)
    tensor([0.0000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000])

    ### Reproducibility with a seed
    >>> clear_label1 = RandomClearLabel(seed=32)
    >>> clear_label2 = RandomClearLabel(seed=32)
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor1 = clear_label1(input_tensor, label_tensor)
    >>> cleared_tensor2 = clear_label2(input_tensor, label_tensor)
    >>> print(torch.equal(cleared_tensor1, cleared_tensor2))
    True
    """

    def __init__(
        self,
        prob: Union[float, int, ne.samplers.Sampler] = 0.5,
        exclude_zero: bool = True,
        seed: int = None,
    ):
        """
        Initialize the `RandomClearLabel` module.

        Parameters
        ----------
        prob : Union[float, int, Sampler], optional
            Probability of any label/region being selected for erasure as determined by iid
            Bernoulli trials, by default 0.5.
        exclude_zero : bool, optional
            Optionally exclude zero (uaually background) from the list of potential regions to clear
            (never clear zero labels), by default True.
        seed : int, optional
            A random seed or sampler to control the randomness of label clearing operations. If
            provided, it ensures reproducibility of the clearing process. By default, None.
        """
        super().__init__()
        self.prob = prob
        self.exclude_zero = exclude_zero
        self.seed = seed

    def forward(
        self,
        input_tensor: torch.Tensor,
        label_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomClearLabel` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Image or tensor to clear.
        label_tensor : torch.Tensor
            Label map corresponding to sampling domain from which to select regions for clearing.

        Returns
        -------
        torch.Tensor
            The modified tensor with specified labels cleared (set to zero). If no labels are
            cleared, the original `input_tensor` is returned unchanged.
        """

        return ne.utils.random_clear_label(
            input_tensor=input_tensor,
            label_tensor=label_tensor,
            prob=self.prob,
            exclude_zero=self.exclude_zero,
            seed=self.seed
        )


class SampleImageFromLabels(nn.Module):
    """
    Generate an image from a label map by uniformly sampling a random intensity for each label.

    `SampleImageFromLabels` identifies all unique integer labels in the `label_tensor`, and assigns
    each a mean intensity to the labeled region in the corresponding output image (`sampled_image`).
    The mean intensity serves as the mean for a noise distribution modeled by `noise_sampler`. The
    variance of the noise model may be a fixed quantity or sampled from another distribution defined
    by `noise_variance`.
    """

    def __init__(
        self,
        mean_sampler: ne.samplers.Sampler = ne.samplers.Uniform(0, 1),
        noise_sampler: ne.samplers.Sampler = ne.samplers.Normal,
        noise_variance: Union[float, int, ne.samplers.Sampler] = 0.25,
    ):
        """
        Initialize the `SampleImageFromLabels` module.

        Parameters
        ----------
        mean_sampler : Sampler
            A `Sampler` from which to draw the mean intensity for each region defined by each label
            in the `label_tensor`. By default, `Uniform(0, 1)`
        noise_sampler : Sampler
            A `Sampler` that is used to model the noise within a particular label/region. The mean
            for the sampler is defined by the mean region intensity (sampled from `mean_sampler`).
            By default, `Normal`.
        noise_variance : float, int, or Sampler
            The variance of the noise model. It can be a fixed quantity (int or float), or a sampled
            quantity in the case a `Sampler` is passed. By default, 0.25.
        """
        super().__init__()
        self.mean_sampler = mean_sampler
        self.noise_sampler = noise_sampler
        self.noise_variance = noise_variance

    def forward(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the sampling operation.

        Parameters
        ----------
        label_tensor : torch.Tensor
            A tensor with batch and channel dimensions containing integer labels defining distinct
            regions.

        Returns
        -------
        torch.Tensor
            A tensor of sampled image intensities with the same shape as `label_tensor`.
        """

        return ne.utils.sample_image_from_labels(
            label_tensor,
            self.mean_sampler,
            self.noise_sampler,
            self.noise_variance
        )
