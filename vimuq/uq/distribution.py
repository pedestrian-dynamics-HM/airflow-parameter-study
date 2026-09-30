import math

import numpy as np


class Bounds:
    def __init__(self, lower_bound: float = 0, upper_bound: float = 1):
        self.lower_bound = lower_bound
        if lower_bound > upper_bound:
            raise ValueError("Lower bound must be smaller than upper bound.")
        self.upper_bound = upper_bound


class Distribution:
    def __init__(self, name: str = None, mean: float = None, std: float = None):
        self.name = name
        self.mean = mean
        self.std = std


class UniformDistribution(Distribution, Bounds):
    def __init__(self, lower_bound=0, upper_bound=1):
        Distribution.__init__(
            self,
            name="unif",
            mean=(lower_bound + upper_bound) / 2.0,
            std=np.sqrt(
                (upper_bound - lower_bound) * (upper_bound - lower_bound) / 12.0
            ),
        )

        Bounds.__init__(self, lower_bound=lower_bound, upper_bound=upper_bound)


class NormalDistribution(Distribution):
    def __init__(self, mean: float = 0, std: float = 1):
        super().__init__(name="norm", mean=mean, std=std)


class TruncatedNormalDistribution(Distribution, Bounds):
    def __init__(
        self,
        mean: float = 0,
        std: float = 1,
        lower_bound: float = -3,
        upper_bound: float = 3,
    ):
        Distribution.__init__(self, name="truncnorm", mean=mean, std=std)
        Bounds.__init__(self, lower_bound=lower_bound, upper_bound=upper_bound)


class LogNormalDistribution(Distribution):
    """Lognormal distribution.
    Parameters are ln-space mean and standard deviation (SALib.util.__init__)
    """

    def __init__(self, mean_ln_space: float = 0, std_ln_space: float = 1):
        Distribution.__init__(
            self,
            name="lognorm",
            mean=mean_ln_space,
            std=std_ln_space,
        )


class Log10UniformDistribution(Distribution, Bounds):
    """Log-uniform distribution.
    Equivalent to uniform(log_{10}(lower), log_{10}(upper))
    Mean and standard deviation are given in log space.

    Defines a uniform distribution in log space but the sampling is done in the
    linear space and then afterwards the samples are transformed to log space.
    """

    BASE = 10

    def __init__(self, lower_bound: float = 1, upper_bound: float = 10):
        if lower_bound > 0:
            lower_log_space = math.log(lower_bound, self.BASE)
        else:
            raise ValueError("Lower bound must be > 0.")
        upper_log_space = math.log(upper_bound, self.BASE)
        Bounds.__init__(self, lower_bound=lower_log_space, upper_bound=upper_log_space)

        Distribution.__init__(
            self,
            name="unif",  # this is required only for the sampling with SALib
            # and Chaospy
            mean=(lower_log_space + upper_log_space) / 2.0,
            std=np.sqrt(
                (upper_log_space - lower_log_space)
                * (upper_log_space - lower_log_space)
                / 12.0
            ),
        )

    def samples_log_to_lin_scale(self, samples: np.array):
        return float(self.BASE) ** samples
