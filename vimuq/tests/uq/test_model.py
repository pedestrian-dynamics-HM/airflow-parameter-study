import shutil
import unittest
import os
from unittest.mock import patch, MagicMock

import pandas as pd

from suqc import PostScenarioChangesBase

from vimuq.tests import _get_head_path
from vimuq.uq.model import Model, VadereModel, PyModel
from vimuq.uq.parameter_space import ParameterSpace


class TestModel(unittest.TestCase):
    def test__check_path_exists_true(self):
        path = os.path.join(".", "exist")
        os.makedirs(path, exist_ok=False)
        self.assertTrue(Model._check_path_exists(path))
        shutil.rmtree(path)

    def test__check_path_exists_throws_error(self):
        path = os.path.join(".", "not", "exist")
        self.assertRaises(FileNotFoundError, Model._check_path_exists, path=path)


class TestVadereModel(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2tests_vadere = os.path.join(
            path2project, "vimuq", "tests", "model", "vadere"
        )

        self._path2scenario = os.path.join(
            path2tests_vadere, "scenarios", "test_queue_uq_fast.scenario"
        )
        self._path2scenario_folder = os.path.join(path2tests_vadere, "scenarios")
        self._path2model = os.path.join(
            path2tests_vadere, "simulator", "test_vadere-console.txt"
        )

    def test___init___rng(self):
        """Check if two model instances get the 'same' random number generator
        in the sense that the resulting random numbers are the same.
        """

        model1 = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi1"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
        )
        model2 = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi2"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
        )

        self.assertEqual(model1.rng.__getstate__(), model2.rng.__getstate__())

    def test___init___use_scenario_files_as_repetitions_default(self):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
        )
        self.assertFalse(model.use_scenario_files_as_repetitions)

    def test___init___use_scenario_files_as_repetitions_set_true(self):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            use_scenario_files_as_repetitions=True,
        )
        self.assertTrue(model.use_scenario_files_as_repetitions)

    def test__set_repetitions_seeds(self):
        repetitions = 10**5
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi2"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            repetitions=repetitions,
        )
        model._set_repetitions_seeds()
        min_signed_int32 = -2147483648
        max_signed_int32 = 2147483647

        self.assertEqual(len(model.repetitions_seeds), repetitions)
        self.assertTrue(min(model.repetitions_seeds) > min_signed_int32)
        self.assertTrue(max(model.repetitions_seeds) < max_signed_int32)
        self.assertTrue(all(isinstance(n, int) for n in model.repetitions_seeds))

    def test_apply_suqc_rand_number_per_repetition_change(self):
        repetitions = 10
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi2"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            repetitions=repetitions,
        )
        model.apply_suqc_rand_number_per_repetition()
        scenario_changes_keys = model.suqc_post_changes._apply_scenario_changes.keys()
        rnd_numbers = model.suqc_post_changes._apply_scenario_changes[
            "random_number"
        ]._fixed_randnr

        self.assertTrue("random_number" in scenario_changes_keys)
        self.assertEqual(len(rnd_numbers), repetitions)

    def test_apply_suqc_rand_number_per_repetition_rng(self):
        repetitions1 = 10
        repetitions2 = repetitions1 + 100

        model1 = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi2"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            repetitions=repetitions1,
        )
        model1.apply_suqc_rand_number_per_repetition()
        rnd1 = model1.suqc_post_changes._apply_scenario_changes[
            "random_number"
        ]._fixed_randnr

        model2 = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi2"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            repetitions=repetitions2,
        )
        model2.apply_suqc_rand_number_per_repetition()
        rnd2 = model2.suqc_post_changes._apply_scenario_changes[
            "random_number"
        ]._fixed_randnr

        self.assertEqual(rnd1[0:repetitions1], rnd2[0:repetitions1])

    def test__init__post_scenario_change(self):
        expected = PostScenarioChangesBase(apply_default=True)
        model1 = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
        )
        model1.apply_suqc_rand_number_per_repetition()

        model2 = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
        )

        self.assertEqual(
            model2.suqc_post_changes._apply_scenario_changes.keys(),
            expected._apply_scenario_changes.keys(),
        )


class TestVadereModelEvaluateDispatch(unittest.TestCase):
    """Tests that evaluate correctly dispatches to run_suqc_scenario_file
    or run_suqc_scenario_folder based on the path2scenario type."""

    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2tests_vadere = os.path.join(
            path2project, "vimuq", "tests", "model", "vadere"
        )

        self._path2scenario_file = os.path.join(
            path2tests_vadere, "scenarios", "test_queue_uq_fast.scenario"
        )
        self._path2scenario_folder = os.path.join(path2tests_vadere, "scenarios")
        self._path2model = os.path.join(
            path2tests_vadere, "simulator", "test_vadere-console.txt"
        )

        self._dummy_input = [{"param_1": 1.0}]
        self._dummy_meta = pd.DataFrame()
        self._dummy_output = {"qoi": pd.DataFrame()}

    @patch.object(VadereModel, "run_suqc_scenario_file")
    def test_evaluate_dispatches_to_scenario_file(self, mock_run_file):
        mock_run_file.return_value = (self._dummy_meta, self._dummy_output)

        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_file,
        )

        model.evaluate(self._dummy_input, path2output="/tmp/test_output")

        mock_run_file.assert_called_once()

    @patch.object(VadereModel, "run_suqc_scenario_folder")
    def test_evaluate_dispatches_to_scenario_folder(self, mock_run_folder):
        mock_run_folder.return_value = (self._dummy_meta, self._dummy_output)

        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_folder,
        )

        model.evaluate(self._dummy_input, path2output="/tmp/test_output")

        mock_run_folder.assert_called_once()

    def test_evaluate_invalid_path_returns_empty(self):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_file,
        )
        model.path2scenario = "/nonexistent/path/to/scenario"

        meta_info, model_output = model.evaluate(self._dummy_input)

        self.assertEqual(meta_info, [])
        self.assertEqual(model_output, {})


class TestRunSuqcScenarioFolder(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2tests_vadere = os.path.join(
            path2project, "vimuq", "tests", "model", "vadere"
        )

        self._path2scenario_folder = os.path.join(path2tests_vadere, "scenarios")
        self._path2model = os.path.join(
            path2tests_vadere, "simulator", "test_vadere-console.txt"
        )

    @patch("vimuq.uq.model.Request")
    @patch("vimuq.uq.model.VadereScenarioCreation")
    @patch("vimuq.uq.model.VadereEnvironmentManager")
    @patch("vimuq.uq.model.UserDefinedSampling")
    def test_run_suqc_scenario_folder_calls_request_run_locally(
        self,
        mock_sampling_cls,
        mock_env_man_cls,
        mock_scenario_creation_cls,
        mock_request_cls,
    ):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_folder,
            run_local=True,
        )

        mock_sampling_instance = MagicMock()
        mock_sampling_instance.multiply_scenario_runs.return_value = (
            mock_sampling_instance
        )
        mock_sampling_instance.points = MagicMock()
        mock_sampling_instance.points.index = MagicMock()
        mock_sampling_instance.points.index.set_levels.return_value = (
            mock_sampling_instance.points.index
        )
        mock_sampling_cls.return_value = mock_sampling_instance

        mock_env_man_cls.create_variation_env.return_value = MagicMock()

        mock_scenario_creation_instance = MagicMock()
        mock_scenario_creation_instance.generate_scenarios.return_value = [MagicMock()]
        mock_scenario_creation_cls.return_value = mock_scenario_creation_instance

        expected_meta = pd.DataFrame({"col": [1]})
        expected_output = pd.DataFrame({"data": [42]})
        mock_request_instance = MagicMock()
        mock_request_instance.run.return_value = (expected_output, expected_meta)
        mock_request_cls.return_value = mock_request_instance

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        model.run_suqc_scenario_folder(
            model_input=dummy_input,
            qoi=["qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        mock_request_instance.run.assert_called_once()

    @patch("vimuq.uq.model.Request")
    @patch("vimuq.uq.model.VadereScenarioCreation")
    @patch("vimuq.uq.model.VadereEnvironmentManager")
    @patch("vimuq.uq.model.UserDefinedSampling")
    def test_run_suqc_scenario_folder_calls_request_remote(
        self,
        mock_sampling_cls,
        mock_env_man_cls,
        mock_scenario_creation_cls,
        mock_request_cls,
    ):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_folder,
            run_local=False,
        )

        mock_sampling_instance = MagicMock()
        mock_sampling_instance.multiply_scenario_runs.return_value = (
            mock_sampling_instance
        )
        mock_sampling_instance.points = MagicMock()
        mock_sampling_instance.points.index = MagicMock()
        mock_sampling_instance.points.index.set_levels.return_value = (
            mock_sampling_instance.points.index
        )
        mock_sampling_cls.return_value = mock_sampling_instance

        mock_env_man_cls.create_variation_env.return_value = MagicMock()

        mock_scenario_creation_instance = MagicMock()
        mock_scenario_creation_instance.generate_scenarios.return_value = [MagicMock()]
        mock_scenario_creation_cls.return_value = mock_scenario_creation_instance

        expected_meta = pd.DataFrame({"col": [1]})
        expected_output = pd.DataFrame({"data": [42]})
        mock_request_instance = MagicMock()
        mock_request_instance.remote.return_value = (expected_output, expected_meta)
        mock_request_cls.return_value = mock_request_instance

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        model.run_suqc_scenario_folder(
            model_input=dummy_input,
            qoi=["qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        mock_request_instance.remote.assert_called_once()

    @patch("vimuq.uq.model.Request")
    @patch("vimuq.uq.model.VadereScenarioCreation")
    @patch("vimuq.uq.model.VadereEnvironmentManager")
    @patch("vimuq.uq.model.UserDefinedSampling")
    def test_run_suqc_scenario_folder_wraps_dataframe_output(
        self,
        mock_sampling_cls,
        mock_env_man_cls,
        mock_scenario_creation_cls,
        mock_request_cls,
    ):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["my_qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_folder,
            run_local=True,
        )

        mock_sampling_instance = MagicMock()
        mock_sampling_instance.multiply_scenario_runs.return_value = (
            mock_sampling_instance
        )
        mock_sampling_instance.points = MagicMock()
        mock_sampling_instance.points.index = MagicMock()
        mock_sampling_instance.points.index.set_levels.return_value = (
            mock_sampling_instance.points.index
        )
        mock_sampling_cls.return_value = mock_sampling_instance

        mock_env_man_cls.create_variation_env.return_value = MagicMock()

        mock_scenario_creation_instance = MagicMock()
        mock_scenario_creation_instance.generate_scenarios.return_value = [MagicMock()]
        mock_scenario_creation_cls.return_value = mock_scenario_creation_instance

        raw_df_output = pd.DataFrame({"data": [1, 2, 3]})
        mock_request_instance = MagicMock()
        mock_request_instance.run.return_value = (raw_df_output, MagicMock())
        mock_request_cls.return_value = mock_request_instance

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        meta_info, model_output = model.run_suqc_scenario_folder(
            model_input=dummy_input,
            qoi=["my_qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        self.assertIsInstance(model_output, dict)
        self.assertIn("my_qoi", model_output)

    @patch("vimuq.uq.model.Request")
    @patch("vimuq.uq.model.VadereScenarioCreation")
    @patch("vimuq.uq.model.VadereEnvironmentManager")
    @patch("vimuq.uq.model.UserDefinedSampling")
    def test_run_suqc_scenario_folder_iterates_scenario_files(
        self,
        mock_sampling_cls,
        mock_env_man_cls,
        mock_scenario_creation_cls,
        mock_request_cls,
    ):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario_folder,
            run_local=True,
        )

        mock_sampling_instance = MagicMock()
        mock_sampling_instance.multiply_scenario_runs.return_value = (
            mock_sampling_instance
        )
        mock_sampling_instance.points = MagicMock()
        mock_sampling_instance.points.index = MagicMock()
        mock_sampling_instance.points.index.set_levels.return_value = (
            mock_sampling_instance.points.index
        )
        mock_sampling_cls.return_value = mock_sampling_instance

        mock_env_man_cls.create_variation_env.return_value = MagicMock()

        mock_scenario_creation_instance = MagicMock()
        mock_scenario_creation_instance.generate_scenarios.return_value = [MagicMock()]
        mock_scenario_creation_cls.return_value = mock_scenario_creation_instance

        mock_request_instance = MagicMock()
        mock_request_instance.run.return_value = ({"qoi": pd.DataFrame()}, MagicMock())
        mock_request_cls.return_value = mock_request_instance

        scenario_files = [
            f for f in os.listdir(self._path2scenario_folder) if f.endswith(".scenario")
        ]
        expected_count = len(scenario_files)

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        model.run_suqc_scenario_folder(
            model_input=dummy_input,
            qoi=["qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        self.assertEqual(
            mock_env_man_cls.create_variation_env.call_count, expected_count
        )
        self.assertEqual(mock_scenario_creation_cls.call_count, expected_count)


class TestRunSuqcScenarioFile(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2tests_vadere = os.path.join(
            path2project, "vimuq", "tests", "model", "vadere"
        )

        self._path2scenario = os.path.join(
            path2tests_vadere, "scenarios", "test_queue_uq_fast.scenario"
        )
        self._path2model = os.path.join(
            path2tests_vadere, "simulator", "test_vadere-console.txt"
        )

    @patch("vimuq.uq.model.DictVariation")
    def test_run_suqc_scenario_file_calls_run_locally(self, mock_dict_variation_cls):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            run_local=True,
        )

        mock_setup = MagicMock()
        mock_setup.run.return_value = (MagicMock(), {"qoi": pd.DataFrame()})
        mock_dict_variation_cls.return_value = mock_setup

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        model.run_suqc_scenario_file(
            model_input=dummy_input,
            qoi=["qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        mock_setup.run.assert_called_once_with(-1)

    @patch("vimuq.uq.model.DictVariation")
    def test_run_suqc_scenario_file_calls_remote(self, mock_dict_variation_cls):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            run_local=False,
        )

        mock_setup = MagicMock()
        mock_setup.remote.return_value = (MagicMock(), {"qoi": pd.DataFrame()})
        mock_dict_variation_cls.return_value = mock_setup

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        model.run_suqc_scenario_file(
            model_input=dummy_input,
            qoi=["qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        mock_setup.remote.assert_called_once_with(-1)

    @patch("vimuq.uq.model.DictVariation")
    def test_run_suqc_scenario_file_wraps_dataframe_output(
        self, mock_dict_variation_cls
    ):
        model = VadereModel(
            parameter_space=ParameterSpace(),
            quantities_of_interest=["my_qoi"],
            path2model=self._path2model,
            path2scenario=self._path2scenario,
            run_local=True,
        )

        raw_df = pd.DataFrame({"data": [1, 2, 3]})
        mock_setup = MagicMock()
        mock_setup.run.return_value = (MagicMock(), raw_df)
        mock_dict_variation_cls.return_value = mock_setup

        dummy_input = [{"param_1": 1.0}]
        jar_command = MagicMock()

        meta_info, model_output = model.run_suqc_scenario_file(
            model_input=dummy_input,
            qoi=["my_qoi.txt"],
            jar_command=jar_command,
            path2output="/tmp/test_output",
        )

        self.assertIsInstance(model_output, dict)
        self.assertIn("my_qoi", model_output)


class TestPyModel(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2tests = os.path.join(path2project, "vimuq", "tests")

        self._path = os.path.join(path2tests, "model", "pymodel", "ishigami.py")

    def test_convert_os_path_to_module(self):
        module_str = PyModel._convert_os_path_to_module(self._path)
        expected = "vimuq.tests.model.pymodel.ishigami"

        self.assertEqual(module_str, expected)

    def test_evaluate_out_format(self):
        dummy_p_space = ParameterSpace()
        qoi = "test_qoi"

        model = PyModel(dummy_p_space, [qoi], self._path)

        samples = [
            {"x1": i * 0.314, "x2": i * 0.314, "x3": i * 0.314} for i in range(-10, 11)
        ]

        _, out = model.evaluate(samples)
        check_sum = out[qoi].to_numpy().sum()

        self.assertTrue("id" in out[qoi].index.names)
        self.assertEqual(out[qoi].to_numpy().shape, (len(samples), 1))
        self.assertAlmostEqual(check_sum, 70.03434808713871, places=8)


if __name__ == "__main__":
    unittest.main()
