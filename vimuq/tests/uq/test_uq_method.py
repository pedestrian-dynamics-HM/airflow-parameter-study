import unittest

# class TestSampledDistributions(unittest.TestCase):
#     def setUp(self):
#         self._p_space = ParameterSpace()
#
#         dists = Distribution.__subclasses__()
#
#         for i in range(len(dists)):
#             self._p_space.add_parameter(Parameter(str(i), dists[i]()))
#             self._expected_means = dists[i]().mean
#             self._expected_stds = dists[i]().std
#
#     def test_sampled_distribution(self):
#         base_samples = 1000
#         method = SALibSaltelliSobolSequence(base_samples)
#         samples = method.sample(self._p_space)
#         output = samples    # that is model f is: f(x) = x
#         output = {"Y": output}
#         uq_output = method.analyze(self._p_space, output)
#         self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
