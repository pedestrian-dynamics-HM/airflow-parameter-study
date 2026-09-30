import unittest

from vimuq.experiments.analysis import run_analysis
from vimuq.tests.imports import v_analysis


class TestRunAnalysis(unittest.TestCase):
    """Test class for testing vimuq.experiments.analysis.run_analysis."""

    def setUp(self):
        self.parser = run_analysis._get_parser()

    def test__get_parser(self):
        args = self.parser.parse_args(v_analysis["argv_f"])
        expected_args = v_analysis["namespace"]

        self.assertEqual(args, expected_args)


if __name__ == "__main__":
    unittest.main()
