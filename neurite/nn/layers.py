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

from typing import Optional, Union, Tuple, List, Literal
import torch
from torch import nn
import torch.nn.functional as F

import neurite as ne
from neurite.samplers import Sampler


class RescaleValues(nn.Module):
    """
    Scale each element of the input tensor by a multiplicative factor.
    """

    def __init__(self, scale_factor: Union[float, int, Sampler]):
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
        mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
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
        mode : str, optional
            Interpolation mode for upsampling. Options include 'nearest', 'linear',
            'bicubic', 'area', and 'nearest-exact'. Default is 'linear'.
        align_corners : bool, optional
            Alignment for "linear", "bilinear", or "trilinear" modes.
        recompute_scale_factor : bool, optional
            If True, recomputes the scale factor for interpolation.
        antialias : bool, default=False
            Applies anti-aliasing if `scale_factor` < 1.0.

        Examples
        --------
        >>> # Get a random tensor ~N(0, 1)
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)

        ### Resize with fixed `scale_factor`
        >>> resize_module = Resize(scale_factor=2)
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor.shape)
        torch.Size([1, 1, 64, 64, 64])

        ### Resize with a sampled `scale_factor`
        >>> resize_module = Resize(scale_factor=Uniform(0.5, 4))
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 74, 74, 74])

        ### Resize to a specific shape
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

    Map continuous values to discrete bins while retaining some smoothness/continuity
    which is parametrized by a softening parameter. It is especially useful in the context of
    machine learning, where it is desirable to have a differentiable version of a quantized
    quantity, allowing for backprop. Hard quantization is non-differentiable and creates gradients
    of zero, making gradient-based optimization impossible.
    """

    def __init__(
        self,
        nb_bins: Union[int, Sampler] = 16,
        softness: Union[float, int, Sampler] = 1.0,
        min_clip: Union[float, int, Sampler] = -float('inf'),
        max_clip: Union[float, int, Sampler] = float('inf'),
        return_log: bool = False,
    ):
        """
        Initialize `SoftQuantize`.

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
        >>> # Make 3D tensor ~N(0, 1).
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> # Initialize & apply the SoftQuantize instance.
        >>> soft_quantizer = SoftQuantize(nb_bins=4, softness=0.5)
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> # Visualize the softly quantized tensor.
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])

        ### Softly quantize with random `nb_bins` and `softness` parameters
        >>> # Get `nb_bins` ~U(3, 32), and `softness` ~U(0.001, 10)
        >>> soft_quantizer = SoftQuantize(nb_bins=RandInt(3, 32), softness=Uniform(0.001, 10))
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])
        """
        super().__init__()
        self.nb_bins = ne.samplers.make_sampler(ne.samplers.Fixed, nb_bins)
        self.softness = ne.samplers.make_sampler(ne.samplers.Fixed, softness)
        self.min_clip = ne.samplers.make_sampler(ne.samplers.Fixed, min_clip)
        self.max_clip = ne.samplers.make_sampler(ne.samplers.Fixed, max_clip)
        self.return_log = return_log

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of `SoftQuantize`.

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
    Calculate the mean squared error (MSE).
    """

    def __init__(self):
        """
        Initialize `MSE`.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between two tensors.

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
        Initialize `GaussianBlur`.

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
        Perform the forward pass of `GaussianBlur`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor, assumed to have 1, 2, or 3 spatial dimensions.

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
    Spatially {subsample, resample} the input tensor.

    This module resamples the input tensor by a factor of `stride` along the specified spatial
    dimension(s) by interleaving dropouts (keeping every `stride`'th element), then upsamples to
    restore it to its original dimensions.
    """

    def __init__(
        self,
        resample_dimension: Union[int, List[int]] = None,
        downsample_stride: Union[int, List[int]] = 2,
        upsample_scale_factor: Union[int, List[int]] = 2,
        mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
        shape: tuple = None,
    ):
        """
        Initialize `Resample`.

        Parameters
        ----------
        resample_dimension : int or list of ints, optional
            The dimension(s) that should be resampled. If None, all dimensions are resampled.
            Default is None.
            dimensions.
        downsample_stride : int or list of ints, optional
            Factor by which to subsample. Default is 2.
        upsample_scale_factor : int, float or list of ints or floats, optional
            Factor by which to upsample. Default is 2.
        mode : str, optional
            Interpolation mode for upsampling. Options include 'nearest', 'linear',
            'bicubic', 'area', and 'nearest-exact'. Default is 'linear'.
        shape : tuple
            Spatial dimensions (without batch or channel dims) to upsample the subsampled tensor
            into.

        Examples
        --------
        ### Subsample with custom stride
        >>> # Make a 2D tensor ~N(0, 1) with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128)
        >>> # Downsample 2x in 1st dim and 4x in second dim. Upsample the same way 
        >>> resampled_tensor = Resample(
        ...    downsample_stride=(2, 4),
        ...    upsample_scale_factor=(2, 4)
        ... )(input_tensor)
        >>> # Ensure spatial dimensions are the same
        >>> print(resampled_tensor.shape)
        torch.Size([1, 1, 128, 128])

        ### Upsample with custom stride and trilinear interpolation
        >>> # Make a 3D tensor ~N(0, 1) with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> # Downsample 2x then upsample 6x
        >>> resampled_tensor = Resample(
        ...    downsample_stride=2,
        ...    upsample_scale_factor=6,
        ...    mode='trilinear'
        ... )(input_tensor)
        >>> # Ensure dimensions are (1, 1, 96, 96, 96)
        >>> print(resampled_tensor.shape)
        torch.Size([1, 1, 96, 96, 96])
        """

        super().__init__()
        self.resample_dimension = resample_dimension
        self.downsample_stride = downsample_stride
        self.upsample_scale_factor = upsample_scale_factor
        self.mode = mode
        self.shape = shape

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the `Resample`.
        """

        return ne.utils.utils.resample(
            input_tensor=input_tensor,
            resample_dimension=self.resample_dimension,
            downsample_stride=self.downsample_stride,
            upsample_scale_factor=self.upsample_scale_factor,
            mode=self.mode,
            shape=self.shape
        )


class RandomCrop(nn.Module):
    """
    Randomly crop the input tensor along allowed dimensions.

    This module randomly selects a subset of the allowed dimensions (excluding `forbidden_dims`)
    and crops each independently by a proportion that is randomly drawn from a distribution. The
    proportion to crop can either be fixed or sampled from a specified distribution. Each allowed
    dimension has a probability `prob` of being cropped based on the results of independent
    Bernoulli trials.
    """

    def __init__(
        self,
        crop_proportion: Union[Sampler, float] = 0.5,
        prob: Union[Sampler, float] = 1,
        forbidden_dims: ne.samplers.Union[Tuple, List] = (0, 1),
        seed: Union[int, Sampler] = None,
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
        clip_min: Union[float, int, Sampler] = 0,
        clip_max: Union[float, int, Sampler] = 1,
        clip_prob: Union[float, int, Sampler] = 0.5,
        seed: Union[int, Sampler] = None,
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
        ### Initialize `RandomClip` and apply it deterministically to a tensor:
        >>> random_clip = RandomClip(clip_min=0.1, clip_max=0.9, clip_prob=0.5)
        >>> input_tensor = torch.randn(3, 3)
        >>> output_tensor = random_clip(input_tensor)
        >>> print(output_tensor)

        ### Clip by sampling min/max bounds from a custom distribution:
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
        Forward pass of `RandomClip`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor to be clipped.

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
    Apply a randomized or deterministic nonlinear gamma scaling operation.

    Adjust the contrast of the input tensor by applying a non-linear scaling operation.
    Each element in the tensor is raised to the power of `gamma`, which is a quantity sampled from
    a distribution.
    """

    def __init__(
        self,
        gamma: Union[float, int, Sampler] = 1.0,
        prob: Union[float, int, Sampler] = 1.0,
        seed: Union[int, Sampler] = None,
    ):
        """
        Initialize `RandomGamma`.

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

        ### Scaling with gamma ~LogNormal(0, 1)
        >>> gamma_sampler = LogNormal(0.5, 1.5)
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
        Forward pass of `RandomGamma`.

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
    Erase regions of an image from randomly selected regions in a label map.

    Identify unique labels within the `label_tensor` and, based on a specified probability,
    designate regions of the `input_tensor` to be erased (set to zero).

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
        prob: Union[float, int, Sampler] = 0.5,
        exclude_zero: bool = True,
        seed: int = None,
    ):
        """
        Initialize `RandomClearLabel`.

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
        Forward pass of `RandomClearLabel`.

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
    Generate an image from a label map by sampling a random intensity for each label.

    Identify all unique integer labels in `label_tensor` and assigns each a mean intensity in the
    corresponding output image (`sampled_image`). The mean intensity serves as the mean for a noise
    distribution modeled by `noise_sampler`. The variance of the noise model may be a fixed quantity
    or sampled from another distribution defined by `noise_variance`.
    """

    def __init__(
        self,
        mean_sampler: Sampler = ne.samplers.Uniform(0, 1),
        noise_sampler: Sampler = ne.samplers.Normal,
        noise_variance: Union[float, int, Sampler] = 0.25,
    ):
        """
        Initialize `SampleImageFromLabels`.

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
        Forward pass of `SampleImageFromLabels`.

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
