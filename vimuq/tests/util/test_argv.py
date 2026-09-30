import unittest

from vimuq.util.argv import split_argument_list

from vimuq.tests.imports import v_experiment


class TestArgv(unittest.TestCase):
    def test_split_argument_list_with_formatting(self):
        argv_list = split_argument_list(v_experiment["argv"])
        expected_argv_list = v_experiment["argv_f"]

        self.assertEqual(argv_list, expected_argv_list)

    def test_split_argument_list_no_formatting(self):
        argv_list = split_argument_list(v_experiment["argv_f"])
        expected_argv_list = v_experiment["argv_f"]

        self.assertEqual(argv_list, expected_argv_list)
