"""
Sampler functionality has moved to neurite-sandbox/etienne_chollet on 2025-11-08.
"""


class Sampler:
    def __init__(self, **theta):
        raise NotImplementedError("Samplers have moved to neurite-sandbox/etienne_chollet.")


class Uniform(Sampler):
    pass


class Fixed(Sampler):
    pass


class Normal(Sampler):
    pass


class Bernoulli(Sampler):
    pass


class Poisson(Sampler):
    pass


class LogNormal(Sampler):
    pass


class RandInt(Sampler):
    pass


def make_sampler(*args, **kwargs):
    raise NotImplementedError("Samplers have moved to neurite-sandbox.")
