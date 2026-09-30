import unittest

from vimuq.uq.distribution import UniformDistribution, Distribution
from vimuq.uq.parameter import Parameter
from vimuq.uq.parameter_space import ParameterSpace


class TestParameterSpace(unittest.TestCase):
    def setUp(self):
        self._parameter = Parameter("p", UniformDistribution())

        self._p_space = ParameterSpace()
        self._expected_p_space = dict()
        self._expected_p_space["num_vars"] = 0
        self._expected_p_space["names"] = list()
        self._expected_p_space["dists"] = list()
        self._expected_p_space["bounds"] = list()

        dists = Distribution.__subclasses__()
        for i in range(len(dists)):
            name = f"p{str(i)}"
            self._p_space.add_parameter(Parameter(name, dists[i]()))

    def test_add_parameter(self):
        parameter_space = ParameterSpace()
        parameter_space.add_parameter(self._parameter)

        self.assertTrue(len(parameter_space.names()) == 1)

    def test_to_salib_format_new_instance(self):
        salib_pspace = self._p_space.to_salib_format()
        self.assertTrue(isinstance(salib_pspace, dict))
        self.assertTrue(isinstance(self._p_space, ParameterSpace))

    def test_to_salib_format(self):
        p_space = ParameterSpace()
        dist = UniformDistribution()
        name = "p"
        p_space.add_parameter(Parameter(name, dist))
        salib_pspace = p_space.to_salib_format()
        expected = {
            "names": [name],
            "num_vars": 1,
            "dists": [dist.name],
            "bounds": [[0, 1]],
        }
        self.assertEqual(salib_pspace, expected)


if __name__ == "__main__":
    unittest.main()
