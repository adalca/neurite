from __future__ import annotations

"""
Random (samplers) for the neurite project.
"""

# Standard library imports
import inspect
from functools import wraps
from typing import Type, Dict, Any, TypeVar, Generator, List, Union, Tuple, Callable

# Third party imports
import torch

__all__ = [
    "register_init_arguments",
    "ensure_list",
    "Sampler",
    "Uniform",
    "Fixed",
    "Normal",
    "Bernoulli",
    "Poisson",
    "LogNormal",
    "RandInt",
    "make_sampler"
]

SamplerType = TypeVar('SamplerType', bound='Sampler')


def register_init_arguments(func: Callable) -> Callable:
    """
    Decorator to register initialization arguments into the instance's `arguments` dict.

    This decorator unpacks a mapping of arbitrary parameters `**theta` and stores each individually.
    If a parameter is an instance of Sampler, it recursively registers its arguments.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original __init__ method
        result = func(self, *args, **kwargs)

        # Initialize the arguments dictionary if it doesn't exist
        if not hasattr(self, 'arguments'):
            self.arguments = {}

        # Get the function's signature
        sig = inspect.signature(func)

        # Bind the passed arguments to the signature
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Extract parameters excluding 'self'
        params = {k: v for k, v in bound.arguments.items() if k != 'self'}

        # Unpack 'theta' if present
        if 'theta' in params:
            theta = params.pop('theta')
            for key, value in theta.items():

                # If the value is a Sampler instance, store its arguments recursively
                if isinstance(value, Sampler):
                    self.arguments[key] = value.serialize()  # Or value.arguments for direct args

                else:
                    self.arguments[key] = value

        # Register the individual arguments
        for key, value in params.items():
            self.arguments[key] = value

        return result

    return wrapper


def ensure_list(x, size=None, crop=True, **kwargs):
    """
    Ensure that an object is a list (of size at last dim).

    Notes
    -----
    - If `x` is a list, nothing is done and `x` is returned as-is (no copy triggered).
    - If `x` is a tuple, it is converted into a list.
    - Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, Generator)):
        x = [x]

    elif not isinstance(x, list):
        x = list(x)

    if size and len(x) < size:
        default = kwargs.get('default', x[-1] if x else None)
        x += [default] * (size - len(x))

    if size and crop:
        x = x[:size]

    return x


class Sampler:
    """
    Base class for random samplers, with a bunch of helpers. This class is for developers of new
    Sampler classes only.
    """

    @register_init_arguments
    def __init__(self, **theta):
        """
        Initializes the Sampler with given (arbitrary) parameters `theta`.

        Parameters
        ----------
        **theta : Any
            Arbitrary keyword arguments representing sampler parameters.
        """
        self.theta = theta

    @classmethod
    def make(
        cls: Type[SamplerType],
        maker_input: Union['Sampler', Dict[str, Any], Tuple[Any, ...], Any]
    ) -> SamplerType:
        """
        Factory method to create a `Sampler` instance from various input types.

        This method allows the creation of `Sampler` objects from existing `Sampler` instances,
        dictionaries, tuples, or other types by intelligently parsing and forwarding the arguments
        to the appropriate constructor.

        Parameters
        ----------
        maker_input : Union[Sampler, Dict[str, Any], Tuple[Any, ...], Any]
            The input from which to create a `Sampler` instance. It can be:
                - An existing `Sampler` instance: Returned as is.
                - A `dict`: Interpreted as keyword arguments for `Sampler` initialization.
                - A `tuple`: Interpreted as positional arguments for `Sampler` initialization.
                - Any other type: Passed as a single positional argument to `Sampler`
                initialization.

        Returns
        -------
        Sampler
            A `Sampler` instance constructed based on the input type.

        Examples
        --------
        >>> # Case 1: Input is an existing Sampler instance
        >>> sampler1 = Sampler(a=1, b=2)
        >>> sampler2 = Sampler.make(sampler1)
        >>> print(sampler1 is sampler2)
        True

        >>> # Case 2: Input is a dictionary
        >>> params = {'a': 10, 'b': 20}
        >>> sampler3 = Sampler.make(params)
        >>> print(sampler3.theta)
        {'a': 10, 'b': 20}

        >>> # Case 3: Input is a tuple
        >>> params_tuple = (100, 200)
        >>> sampler4 = Sampler.make(params_tuple)
        >>> print(sampler4.theta)
        {'min': 100, 'max': 200}

        >>> # Case 4: Input is another type (e.g., int)
        >>> sampler5 = Sampler.make(300)
        >>> print(sampler5.theta)
        {'min': 0, 'max': 300}
        """
        if isinstance(maker_input, Sampler):
            # If it's a `Sampler`, return as-is
            return maker_input

        elif isinstance(maker_input, dict):
            # Interpret dict as kwargs
            return cls(**maker_input)

        elif isinstance(maker_input, tuple):
            # Interpret as positional args
            if len(maker_input) > 2:
                raise TypeError(
                    f"make expected at most 2 positional arguments in tuple, got {len(maker_input)}"
                )
            return cls(*maker_input)

        else:
            # Pass as single positional arg
            return cls(maker_input)

    def _ensure_same_length(
        self,
        theta: Dict[str, Any],
        nsamples: int = None
    ) -> Dict[str, List[Any]]:
        """
        Ensures that all parameter lists in `theta` have the same length.

        If `nsamples` is specified, it extends or truncates each parameter list in `theta` to match
        `nsamples`. If `nsamples` is not specified, it determines the maximum length among the
        parameter lists and adjusts all lists to that length.

        Parameters
        ----------
        theta : Dict[str, Any]
            A dictionary of parameters where values can be single items or lists/tuples.
        nsamples : int, optional
            The desired length for all parameter lists. If `None`, the maximum existing list length
            is used.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary with all parameter lists adjusted to have the same length.
        """
        # `theta` needs to be a dict
        theta = dict(theta)

        # Ensure all parameter lists have a specified or congruent number of elements.
        if nsamples:
            # We need to bring all params to the known length of `nsamples`
            # Validating `nsamples`
            if not isinstance(nsamples, int) or nsamples <= 0:
                raise ValueError(f"`num_samples` must be a positive integer, got {nsamples}")

            # Make sure all param lists have length `nsamples`
            for param_name, param_value in theta.items():
                theta[param_name] = ensure_list(param_value, nsamples)

        else:
            # We need to determine the max len among all parameter lists (since it's not specified)
            max_length = 0

            for param_value in theta.values():
                if isinstance(param_value, (list, tuple)):
                    # If list, check if len exceeds `max_length`
                    max_length = max(max_length, len(param_value))

            if max_length == 0:
                # All parameters are single values; convert them to single-element lists
                for param_name, param_value in theta.items():
                    theta[param_name] = ensure_list(param_value)

            else:
                # Adjust all parameters to the maximum length found
                for param_name, param_value in theta.items():
                    theta[param_name] = ensure_list(param_value, max_length)

        return theta

    @classmethod
    def map(cls, fn: Any, *values: Any, n: int = None) -> Any:
        """
        Applies a function `fn` across multiple lists of values.

        If `n` is specified, each input list is extended or truncated to length `n` before mapping.

        Parameters
        ----------
        fn : Callable
            The function to apply to the mapped values.
        *values : Any
            Multiple lists or single values to be mapped.
        n : int, optional
            The number of samples. If specified, each list in `values` is adjusted to this length.

        Returns
        -------
        Any
            The result of applying `fn` to the mapped values. This could be a list or a single
            value.

        Examples
        --------
        >>> import random
        >>> Sampler.map(random.gauss, [1, 2, 3], [1, 9, 1])
        [0.9751766007463074, 2.4656489254905654, 0.33309233499125224]
        >>> Sampler.map(random.uniform, [0, 1, 2], [10, 20, 100])
        [2.3064614400086403, 7.6623252443254, 99.70018921460807]
        """
        if n:
            values = tuple(ensure_list(value, n) for value in values)

        if isinstance(values[0], list):
            # Apply function to each argument in the list independently
            return [fn(*args) for args in zip(*values)]

        else:
            return fn(*values)

    def __getattr__(self, item: str) -> Any:
        """
        Allows dynamic access to parameters stored in `theta`.

        If an attribute is not found through the normal mechanisms, this method is called.
        It attempts to retrieve the attribute from the `theta` dictionary, ensuring all parameter
        lists have the same length.

        Parameters
        ----------
        item : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The value of the requested attribute from `theta`.

        Examples
        --------
        >>> sampler = Sampler(a=1, b=2)
        >>> sampler.a
        1
        >>> sampler.c
        AttributeError: c
        """
        theta = self.__getattribute__('theta')

        if item in theta:
            return theta[item]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, item: str, value: Any) -> None:
        """
        Allows dynamic setting of parameters stored in `theta`.

        If the attribute exists in `theta`, it updates its value. Otherwise, it sets the attribute
        using the default `__setattr__` behavior.

        Parameters
        ----------
        item : str
            The name of the attribute to set.
        value : Any
            The value to assign to the attribute.
        """
        if item == 'theta':
            return super().__setattr__(item, value)

        if 'theta' in self.__dict__ and item in self.theta:
            self.theta[item] = value

        else:
            return super().__setattr__(item, value)

    def __call__(self, n=None, **backend):
        """
        Generates samples based on the sampler's parameters.

        This method handles the interpretation of the `n` parameter and delegates
        the actual sampling to the `_sample` method, which must be implemented by
        subclasses.

        Parameters
        ----------
        n : Union[int, List[int], Tuple[int, ...]], optional
            Specifies the number or shape of samples to generate.

        Returns
        -------
        Union[Any, List[Any], torch.Tensor]
            The generated samples.
        """
        # Determine the sample shape based on `n`
        if n is None:
            shape = [1]

        elif isinstance(n, int):
            if n <= 0:
                raise ValueError("`n` must be a positive integer.")
            shape = [n]

        elif isinstance(n, (list, tuple)):
            if not all(isinstance(dim, int) and dim > 0 for dim in n):
                raise ValueError("All elements in `n` must be positive integers.")
            shape = list(n)

        else:
            raise TypeError("`n` must be `None`, an `int`, or a list/tuple of ints.")

        # Generate samples using the subclass's _sample method
        samples = self._sample(shape=shape, **backend)

        # Format the output based on the input `n`
        if n is None:
            return samples.view(-1)[0].item()

        elif isinstance(n, int):
            return samples.view(-1).tolist()

        else:
            return samples

    def _sample(self, shape, **backend):
        """
        Abstract method to generate samples. Must be implemented by subclasses.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor.
        """
        raise NotImplementedError(
            "The _sample method must be implemented by subclasses of Sampler."
        )

    def serialize(self) -> dict:
        """
        Serializes the object's state into a dictionary.

        This method captures key attributes of the object and metadata about its
        class and module for purposes such as taxonomy, reconstruction, or debugging.

        Notes
        -----
        The `type` field captures the name of the immediate parent class, which
        can be useful for hierarchical categorization. The `module` and `qualname`
        fields ensure the object's origin can be traced and reconstructed if
        necessary.
        """
        state_dict = {
            # The qualified name of the class (for reconstruction purposes)
            'qualname': self.__class__.__name__,

            # Parent class, for more broad taxonomy/snapshot view
            'parent': type(self).__bases__[0].__name__,

            # The module that the sample may be found in (and reconstructed from)
            'module': self.__module__,

            # The sampler's parameters
            'theta': self.arguments,
        }

        return state_dict


class Uniform(Sampler):
    """
    Sampler that generates uniformly distributed floating-point numbers within a specified range.
    """

    def __init__(
        self,
        min_val: Union[int, float] = 0.0,
        max_val: Union[int, float] = 1.0
    ):
        """
        Sampler that generates uniformly distributed floating-point numbers within a specified
        range.

        This sampler produces samples from a uniform distribution over the interval
        `[min_val, max_val]`. It leverages PyTorch's random number generation capabilities to create
        the samples.

        Parameters
        ----------
        min_val : float or int, optional
            The lower bound of the sampling range (inclusive). Default is 0.0.
        max_val : float or int, optional
            The upper bound of the sampling range (inclusive). Default is 1.0.

        Attributes
        ----------
        theta : Dict[str, Any]
            Dictionary storing the sampling parameters (`min_val` and `max_val`).

        Returns
        -------
        Union[int, List[int], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        >>> # Instantiate `Uniform` with default range [0.0, 1.0]
        >>> sampler = Uniform()
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        0.5372732281684875

        >>> # Instantiate `Uniform` with custom range [5, 10]
        >>> sampler = Uniform(min_val=5, max_val=10)
        >>> # Generate 5 samples as a list
        >>> samples = sampler(n=5)
        >>> print(samples)
        [7.829, 5.123, 9.456, 6.789, 8.012]

        >>> # Instantiate `Uniform` and generate a tensor of samples
        >>> sampler = Uniform(min_val=-2.0, max_val=2.0)
        >>> Sample a tensor with shape (3, 2)
        >>> tensor_samples = sampler(n=[3, 2])
        >>> print(tensor_samples)
        tensor([[-1.2345,  0.5678],
                [ 1.8901, -0.1234],
                [ 0.4567,  1.2345]])
        """
        super().__init__(min_val=min_val, max_val=max_val)

        # Validate parameters
        if not isinstance(min_val, (int, float)):
            # min_val is not an integer or floating point.
            raise TypeError(f"`min_val` must be an int or float, got {type(min_val).__name__}")

        if not isinstance(max_val, (int, float)):
            # max_val is not an integer or floating point.
            raise TypeError(f"`max_val` must be an int or float, got {type(max_val).__name__}")

        if max_val <= min_val:
            # max_val is less than min_val.
            raise ValueError("`max_val` must be greater than `min_val`.")

    def _sample(self, shape: list, **backend) -> torch.Tensor:
        """
        Abstract method to sample from `Uniform`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor.
        """
        min_val = self.theta.get('min_val')
        max_val = self.theta.get('max_val')

        return torch.rand(*shape, **backend) * (max_val - min_val) + min_val


class Fixed(Sampler):
    """
    Sampler that generates a fixed constant value.
    """

    def __init__(self, value: Any):
        """
        Sampler that generates a fixed constant value.

        This sampler always returns the same fixed value specified during initialization.
        It can generate single scalar samples, lists of the fixed value, or tensors filled with the
        fixed value based on the input parameter `n` in __call__.

        Parameters
        ----------
        value : Any
            The fixed value to return when sampling.

        Attributes
        ----------
        theta : Dict[str, Any]
            Dictionary storing the sampling parameter `value`.

        Returns
        -------
        Union[int, List[int], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        ### Single realization
        >>> # Instantiate `Fixed` with a fixed value of 5
        >>> sampler = Fixed(value=5)
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        5

        ### Listed/repeated realizations
        >>> # Instantiate `Fixed` with a fixed value of 3.14
        >>> sampler = Fixed(value=3.14)
        >>> # Generate 4 samples as a list
        >>> samples = sampler(n=4)
        >>> print(samples)
        [3.14, 3.14, 3.14, 3.14]

        ### Tensor filled with realizations
        >>> # Instantiate `Fixed` with a fixed value of -1
        >>> sampler = Fixed(value=-1)
        >>> # Generate a tensor of samples with shape (2, 3)
        >>> tensor_samples = sampler(n=[2, 3])
        >>> print(tensor_samples)
        tensor([[-1, -1, -1],
                [-1, -1, -1]])
        """
        super().__init__(value=value)

    def _sample(self, shape: list, **backend) -> torch.Tensor:
        """
        Abstract method to sample from `Fixed`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor.
        """
        # Extract `value`
        value = self.theta.get('value')

        return torch.full(shape, fill_value=value, **backend)


class Normal(Sampler):
    """
    Sampler that generates normally distributed floating-point numbers based on mean and variance.
    """

    def __init__(
        self,
        mean: Union[int, float] = 0.0,
        variance: Union[int, float] = 1.0
    ):
        """
        Sampler that generates normally distributed floating-point numbers based on mean and
        variance.

        This sampler produces samples from a normal (Gaussian) distribution defined by specified
        `mean` and `variance` parameters.

        Parameters
        ----------
        mean : float or int, optional
            The mean of the normal distribution. Default is 0.0.
        variance : float or int, optional
            The variance of the normal distribution. Must be positive.
            Default is 1.0.

        Attributes
        ----------
        theta : Dict[str, Any]
            Dictionary storing the sampling parameters (`mean` and `variance`).

        Returns
        -------
        Union[int, List[int], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        ### Default parameters
        >>> # Instantiate `Normal` with default parameters
        >>> sampler = Normal()
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        0.1234567890123456

        ### Custom mean and variance
        >>> # Instantiate `Normal` with custom mean and variance
        >>> sampler = Normal(mean=5.0, variance=4.0)
        >>> # Generate 5 samples as a list
        >>> samples = sampler(5)
        >>> print(samples)
        [4.5678, 6.1234, 5.7890, 3.4567, 7.8901]

        ### Sampling a tensor od realizations
        >>> # Instantiate `Normal` and generate a tensor of realizations
        >>> sampler = Normal(mean=-1.0, variance=0.25)
        >>> # Sample a tensor with shape (2, 3)
        >>> tensor_samples = sampler([2, 3])
        >>> print(tensor_samples)
        tensor([[-1.2345, -0.5678, -1.8901],
                [-0.1234, -1.6789, -1.3456]])
        """
        super().__init__(mean=mean, variance=variance)

        # Validate parameters
        if not isinstance(mean, (int, float)):
            raise TypeError(f"`mean` must be an int or float, got {type(mean).__name__}")

        if not isinstance(variance, (int, float)):
            raise TypeError(f"`variance` must be an int or float, got {type(variance).__name__}")

        if variance <= 0:
            raise ValueError("`variance` must be positive.")

    def _sample(self, shape, **backend):
        """
        Abstract method to sample from `Normal`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor.
        """
        # Extract mean and var
        mean = self.theta.get('mean', 0.0)
        variance = self.theta.get('variance', 1.0)

        # Calculate sigma
        sigma = variance ** 0.5
        distribution = torch.distributions.Normal(loc=mean, scale=sigma)

        return distribution.sample(shape)


class Bernoulli(Sampler):
    """
    Sampler that generates binary outcomes (0 or 1) based on a specified probability `p`.
    """

    def __init__(self, p: float = 0.5):
        """
        This sampler generates samples from a Bernoulli distribution, where `p` is the
        probability of success (resulting in 1), and `1 - p` is the probability of
        failure (resulting in 0).

        Parameters
        ----------
        p : float, optional
            Probability of realizing a success (i.e., the probability of a 1) from the Bernoulli
            distribution. By default, 0.5. Must be in the range [0, 1].

        Attributes
        ----------
        theta : Dict[str, Any]
            Dictionary storing the sampling parameter `p`.

        Returns
        -------
        Union[int, List[int], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        ### Single realization
        >>> # Instantiate `Bernoulli` with p=0.7
        >>> sampler = Bernoulli(p=0.7)
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        1

        ### Multiple realizations as a list
        >>> # Generate 5 samples as a list
        >>> samples = sampler(n=5)
        >>> print(samples)
        [1, 0, 1, 1, 0]

        ### Tensor of realizations
        >>> # Generate a tensor of samples with shape (2, 3)
        >>> tensor_samples = sampler(n=[2, 3])
        >>> print(tensor_samples)
        tensor([[1, 0, 1],
                [1, 1, 0]])
        """
        super().__init__(p=p)

        # Validate parameter
        if not isinstance(p, (int, float)):
            raise TypeError(f"`p` must be an int or float, got {type(p).__name__}")

        if not 0 <= p <= 1:
            raise ValueError("`p` must be between 0 and 1 inclusive.")

    def _sample(self, shape, **backend):
        """
        Generates samples from a Bernoulli distribution based on probability `p`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor of 0s and 1s.
        """
        # Extract parameter
        p = self.theta.get('p')
        distribution = torch.distributions.Bernoulli(probs=p)
        samples = distribution.sample(shape).int()

        return samples


class Poisson(Sampler):
    """
    Sampler that generates samples from a Poisson distribution with a specified rate parameter.
    """

    def __init__(self, rate: float = 1.0):
        """
        Sampler that generates samples from a Poisson distribution with a specified rate parameter.

        This sampler produces samples from a Poisson distribution, where `rate` is the average rate
        of occurrence of the event per interval.
        Initialize the `Poisson` sampler with a specified rate parameter `rate`.

        Parameters
        ----------
        rate : float, optional
            The rate parameter (λ) of the Poisson distribution. Must be a positive value.
            Default is 1.0.

        Returns
        -------
        Union[int, List[int], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        ### Single realization
        >>> # Instantiate `Poisson` with rate=2.5
        >>> sampler = Poisson(2.5)
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        3

        ### Multiple realizations as a list
        >>> # Generate 5 samples as a list
        >>> samples = sampler(n=5)
        >>> print(samples)
        [2, 4, 3, 1, 5]

        ### Tensor of realizations
        >>> # Generate a tensor of samples with shape (2, 3)
        >>> tensor_samples = sampler([2, 3])
        >>> print(tensor_samples)
        tensor([[2, 3, 1],
                [4, 2, 5]])
        """
        super().__init__(rate=rate)

        # Validate parameter
        if not isinstance(rate, (int, float)):
            raise TypeError(f"`rate` must be an int or float, got {type(rate).__name__}")

        if rate <= 0:
            raise ValueError("`rate` must be a positive value.")

    def _sample(self, shape, **backend):
        """
        Generates samples from a Poisson distribution based on `rate`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor of integers.
        """
        # Extract parameter
        rate = self.theta.get('rate')
        distribution = torch.distributions.Poisson(rate=rate)
        samples = distribution.sample(shape)

        return samples.int()


class LogNormal(Sampler):
    """
    Sampler that generates log-normally distributed floating-point numbers based on mean and
    variance.
    """

    def __init__(
        self,
        mean: Union[int, float] = 0.0,
        variance: Union[int, float] = 1.0
    ):
        """
        Sampler that generates log-normally distributed floating-point numbers based on mean and
        variance.

        This sampler produces samples from a log-normal distribution, which is the distribution of a
        random variable whose logarithm is normally distributed. It is defined by the parameters
        `mean` and `variance`, where `mean` is the mean of the underlying normal distribution, and
        `variance` is the standard deviation.

        Parameters
        ----------
        mean : float or int, optional
            The mean (`mu`) of the underlying normal distribution. Default is 0.0.
        variance : float or int, optional
            The standard deviation (`variance`) of the underlying normal distribution. Must be
            positive. Default is 1.0.

        Attributes
        ----------
        theta : Dict[str, Any]
            Dictionary storing the sampling parameters (`mean` and `variance`).

        Returns
        -------
        Union[float, List[float], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        ### Single realization
        >>> # Instantiate `LogNormal` with default parameters
        >>> sampler = LogNormal()
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        1.234567890123456

        ### Multiple realizations as a list
        >>> # Instantiate `LogNormal` with custom mean and variance
        >>> sampler = LogNormal(mean=0.5, variance=0.75)
        >>> # Generate 5 samples as a list
        >>> samples = sampler(n=5)
        >>> print(samples)
        [1.6487, 2.0138, 1.2840, 1.5623, 2.3456]

        ### Tensor of realizations
        >>> # Instantiate `LogNormal` and generate a tensor of samples
        >>> sampler = LogNormal(mean=-0.5, variance=0.5)
        >>> # Generate a tensor with shape (2, 3)
        >>> tensor_samples = sampler(n=[2, 3])
        >>> print(tensor_samples)
        tensor([[0.6065, 1.2840, 0.7788],
                [1.1052, 0.6065, 1.6487]])
        """
        super().__init__(mean=mean, variance=variance)

        # Validate parameters
        if not isinstance(mean, (int, float)):
            raise TypeError(f"`mean` must be an int or float, got {type(mean).__name__}")

        if not isinstance(variance, (int, float)):
            raise TypeError(f"`variance` must be an int or float, got {type(variance).__name__}")

        if variance <= 0:
            raise ValueError("`variance` must be a positive value.")

    def _sample(self, shape: list, **backend) -> torch.Tensor:
        """
        Generates samples from a LogNormal distribution based on `mean` and `variance`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor of positive floating-point numbers.
        """
        # Extract parameters
        mean = self.theta.get('mean', 0.0)
        variance = self.theta.get('variance', 1.0)

        # Calculate sigma
        sigma = variance ** 0.5

        # Create LogNormal distribution
        distribution = torch.distributions.LogNormal(loc=mean, scale=sigma)

        # Sample from the distribution
        samples = distribution.sample(shape, **backend)

        return samples


class RandInt(Sampler):
    """
    Sampler that generates uniformly distributed random integers within a specified range.
    """

    def __init__(
        self,
        low: int = 0,
        high: int = 10
    ):
        """
        Sampler that generates uniformly distributed random integers within a specified range.

        This sampler produces samples from a uniform integer distribution over the interval
        `[low, high)`, where `low` and `high` are inclusive. It leverages PyTorch's
        random number generation capabilities to create the samples.

        Parameters
        ----------
        low : int, optional
            The lower bound of the sampling range (inclusive). Default is 0.
        high : int, optional
            The upper bound of the sampling range (exclusive). Must be greater than `low`.
            Default is 10.

        Attributes
        ----------
        theta : Dict[str, Any]
            Dictionary storing the sampling parameters (`low` and `high`).

        Returns
        -------
        Union[int, List[int], torch.Tensor]
            The sampled realizations, varying in type based on `n` to __call__.

        Examples
        --------
        ### Single realization
        >>> # Instantiate `RandInt` with default range [0, 10)
        >>> sampler = RandInt()
        >>> # Single scalar sample
        >>> sample = sampler()
        >>> print(sample)
        7

        ### Multiple realizations as a list
        >>> # Instantiate `RandInt` with custom range [5, 15)
        >>> sampler = RandInt(low=5, high=15)
        >>> # Generate 5 samples as a list
        >>> samples = sampler(n=5)
        >>> print(samples)
        [12, 5, 9, 14, 7]

        ### Tensor of realizations
        >>> # Instantiate `RandInt` and generate a tensor of samples
        >>> sampler = RandInt(low=100, high=200)
        >>> # Generate a tensor with shape (2, 3)
        >>> tensor_samples = sampler(n=[2, 3])
        >>> print(tensor_samples)
        tensor([[150, 123, 178],
                [199, 101, 156]])
        """
        super().__init__(low=low, high=high)

        # Validate parameters
        if not isinstance(low, int):
            raise TypeError(f"`low` must be an int, got {type(low).__name__}")

        if not isinstance(high, int):
            raise TypeError(f"`high` must be an int, got {type(high).__name__}")

        if high < low:
            raise ValueError("`high` must be greater than `low`.")

    def _sample(self, shape: list, **backend) -> torch.Tensor:
        """
        Generates samples from a uniform integer distribution based on `low` and `high`.

        Parameters
        ----------
        shape : List[int]
            The shape of the samples to generate.
        **backend : Any
            Additional keyword arguments for backend configurations (e.g., device, dtype).

        Returns
        -------
        torch.Tensor
            The generated samples as a tensor of integers within the range `[low, high)`.
        """
        # Extract parameters
        low = self.theta.get('low', 0)
        high = self.theta.get('high', 10)

        # Generate samples using torch.randint
        samples = torch.randint(low=low, high=high+1, size=shape, **backend)

        return samples


def _get_sampler_checking_info(sampler_to_check: Union[type, Sampler]):
    """
    Retrieve the sampler class and the number of required arguments for its constructor.

    This function inspects the given sampler (which can be either a class or an instance of a
    sampler) and determines the corresponding sampler class. It then uses Python's introspection to
    count the number of parameters in the sampler's constructor.

    Parameters
    ----------
    sampler_to_check : type or Sampler
        The sampler to inspect. This can either be a sampler class or an instance of a sampler.

    Returns
    -------
    tuple
        A tuple containing:
            - `sampler_cls`: The sampler class corresponding to the input.
            - `max_number_of_arguments`: The max number of parameters in the sampler's constructor.
    """

    # Determine if sampler is a class or an instance and get the class
    if isinstance(sampler_to_check, type):
        sampler_cls = sampler_to_check

    elif isinstance(sampler_to_check, Sampler):
        sampler_cls = sampler_to_check.__class__

    # Use inspect.signature to obtain the constructor's parameters
    inspected_arguments = inspect.signature(sampler_cls)

    # Introspection to get the parameters/init args
    max_number_of_arguments = len(dict(inspected_arguments.parameters))

    return sampler_cls, max_number_of_arguments


def make_sampler(default_sampler: Sampler = Fixed, maker_input: Any = None):
    """
    Construct a sampler based on the provided default sampler and optionally, the maker input.

    This function dynamically constructs a sampler instance using the `default_sampler` class or
    instance, and the provided `maker_input`. Depending on the type and content of `maker_input`,
    it validates and  processes the input, then instantiates or returns an appropriate sampler.

    Parameters
    ----------
    default_sampler : ne.Sampler, optional
        The default sampler class or instance to be used if no valid sampler is provided through
        `maker_input`. Defaults to `ne.Fixed`.

    maker_input : list, tuple, dict, ne.Sampler, int, float, type, or None, optional
        The input used for constructing or validating the sampler. This input can be:
            - A list or tuple of arguments for the sampler constructor.
            - A dict of keyword arguments for the sampler constructor.
            - An instance of `ne.Sampler`, in which case it is returned as is.
            - An int or float to be passed as a single argument.
            - A type object to be rejected if both inputs are types.
            - None, which triggers a special case if `default_sampler` is a type.

    Returns
    -------
    ne.Sampler
        An instance of a sampler constructed with the provided input.

    Examples
    --------
    >>> # Case 1: Input is an existing instance of a Sampler
    >>> sampler1 = ne.Uniform(min_val=0, max_val=1)
    >>> sampler2 = ne.make_sampler(sampler1)
    >>> print(sampler1 is sampler2)
    True

    >>> # Case 2: Input is an existing instance of Sampler, but there is already a default sampler
    >>> new_sampler = ne.Poisson(37)
    >>> sampler = ne.make_sampler(ne.Uniform(-1, 0), new_sampler)
    >>> print(sampler.serialize())
    {
        'qualname': 'Poisson',
        'parent': 'Sampler',
        'module': 'neurite.pytorch_backend.samplers',
        'theta': {'rate': 37}
    }

    >>> # Case 3: Input is a dictionary
    >>> sampler_theta_params = {"mean": 0, "variance": 1}
    >>> sampler1 = ne.make_sampler(ne.LogNormal, sampler_theta_params)
    >>> print(sampler1.theta)
    {'mean': 0, 'variance': 1}

    >>> # Case 4: Input is a tuple
    >>> sampler_theta_params_tuple = (1, 2)
    >>> sampler = ne.make_sampler(ne.Normal, sampler_theta_params_tuple)
    >>> print(sampler.theta)
    {'mean': 1, 'variance': 2}
    """

    # Get the class of the sampler and the max number of arguments for the sampler
    default_sampler_cls, max_number_of_arguments = _get_sampler_checking_info(default_sampler)

    # Logic for `maker_input` if it is a collection
    if isinstance(maker_input, (list, tuple)):

        # If the list/tuple has a single element, simplify by extracting that element
        if len(maker_input) == 1:
            maker_input = maker_input[0]

        elif len(maker_input) <= max_number_of_arguments:
            # If it's a valid list/tuple, construct the class by unpacking the collection
            return default_sampler_cls(*maker_input)

        else:
            # Too many arguments have been provided
            raise ValueError(
                f"`maker_input` can only have at most {max_number_of_arguments} argument(s)!"
            )

    # Process maker_input when it is a dictionary
    elif isinstance(maker_input, dict):

        # Unpack the kwarg mapping to construct instance from class
        return default_sampler_cls(**maker_input)

    elif isinstance(maker_input, Sampler):
        # But! If maker_input is already a sampler instance, return it directly
        return maker_input

    elif maker_input is None:

        if isinstance(default_sampler, type):

            # Cannot use None if default_sampler is not a sampler object
            raise ValueError(
                "`default_sampler` must be an instance of a (Sampler) class if you want to pass "
                "null input to `make_sampler()`"
            )

        else:
            return default_sampler

    elif isinstance(maker_input, type) and isinstance(default_sampler, type):

        # If both maker_input and default_sampler are types, raise an error
        raise ValueError(
            "Neither the default sampler nor maker input are sampler objects! I can't sample like "
            "this!"
        )

    # Process maker_input when it is a numeric type (int or float)
    if isinstance(maker_input, (int, float)):

        if max_number_of_arguments == 0:
            #  If no arguments are expected, the numeric input is invalid
            raise ValueError(
                f"The input to `make_sampler` must have {max_number_of_arguments} arguments"
            )

        else:

            # Use the numeric input as a single argument for instantiation
            return default_sampler_cls(maker_input)
