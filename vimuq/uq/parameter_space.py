import chaospy as cp

from vimuq.uq.distribution import (
    LogNormalDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
    Log10UniformDistribution,
)
from vimuq.uq.parameter import Parameter


class ParameterSpace:
    """Joint model input parameters"""

    def __init__(self):
        self.parameters = list()
        self.dimension = 0

    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter)
        self.dimension = len(self.parameters)

    def names(self):
        return [element.name for element in self.parameters]

    def shapes(self):
        return [element.dist.name for element in self.parameters]

    def to_salib_format(self):
        """Transforms ParameterSpace to SALib compatible dictionary"""

        problem = dict()
        problem["names"] = self.names()
        problem["num_vars"] = len(problem["names"])
        problem["dists"] = self.shapes()

        # From SALib.util._nonuniform_scale_samples:
        # bounds entry indicates either parameter bounds or sample-specific
        # metadata:
        # unif:         [lower bound, upper bound]
        # norm:         [mean, standard deviation]
        # truncnorm:    [lower bound, upper bound, mean, std]
        # triang:       [lower bound, percentage of peak width, upper bound]
        #               currently not supported in vimuq
        # lognorm:      [ln-space mean, ln-space standard deviation]
        # logunif:      actually not compatible with SALib, but this is handled
        #               manually
        problem["bounds"] = []
        for p in self.parameters:
            dist = p.dist
            if isinstance(dist, UniformDistribution):
                b_temp = [dist.lower_bound, dist.upper_bound]
            elif isinstance(dist, NormalDistribution):
                b_temp = [dist.mean, dist.std]
            elif isinstance(dist, TruncatedNormalDistribution):
                b_temp = [dist.lower_bound, dist.upper_bound, dist.mean, dist.std]
            elif isinstance(dist, LogNormalDistribution):
                b_temp = [dist.mean, dist.std]
            elif isinstance(dist, Log10UniformDistribution):
                b_temp = [dist.lower_bound, dist.upper_bound]
            else:
                raise ValueError(
                    "Distribution of shape {} is (currently) not supported.".format(
                        dist.name
                    )
                )

            problem["bounds"].append(b_temp)

        return problem

    def to_chaospy_joint_distribution(self):
        """Transforms ParameterSpace to ChaosPy compatible joint distribution"""
        dists = []
        for p in self.parameters:
            if isinstance(p.dist, UniformDistribution):
                dists.append(
                    cp.Uniform(lower=p.dist.lower_bound, upper=p.dist.upper_bound)
                )
            elif isinstance(p.dist, NormalDistribution):
                dists.append(cp.Normal(mu=p.dist.mean, sigma=p.dist.std))
            elif isinstance(p.dist, TruncatedNormalDistribution):
                dists.append(
                    cp.TruncNormal(
                        lower=p.dist.lower_bound,
                        upper=p.dist.upper_bound,
                        mu=p.dist.mean,
                        sigma=p.dist.std,
                    )
                )
            elif isinstance(p.dist, Log10UniformDistribution):
                dists.append(
                    cp.Uniform(lower=p.dist.lower_bound, upper=p.dist.upper_bound)
                )
            # ToDo elif isinstance(p.dist, LogNormalDistribution):
            else:
                raise ValueError(
                    "Distribution of shape {} is (currently) not supported.".format(
                        p.dist.name
                    )
                )

        chaospy_jdist = cp.J(*dists)

        return chaospy_jdist
