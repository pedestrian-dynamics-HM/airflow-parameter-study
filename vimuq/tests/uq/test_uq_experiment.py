import os
import unittest

import numpy as np

from vimuq.tests import get_sim_output_data_frame, _get_head_path
from vimuq.uq.model import Model
from vimuq.uq.parameter_space import ParameterSpace
from vimuq.uq.uq_experiment import UQExperiment
from vimuq.uq.uq_method import UQMethod


class TestUQExperiment(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2model = os.path.join(
            path2project,
            "vimuq",
            "tests",
            "model",
            "vadere",
            "simulator",
            "test_vadere-console.txt",
        )
        self.dummy_experiment = UQExperiment(
            model=Model(
                ParameterSpace(),
                quantities_of_interest=[],
                path2model=path2model,
            ),
            method=UQMethod(),
            path2output="",
        )

    def test_remove_redundant_index_levels_with_opt_indices(self):
        t = 10
        p = 2
        output, _ = get_sim_output_data_frame(n_peds=p, n_tsteps=t, redundant_idx=True)
        output_ = UQExperiment.remove_redundant_index_levels(output)

        expected, _ = get_sim_output_data_frame(
            n_peds=p, n_tsteps=t, redundant_idx=False
        )

        self.assertTrue(output_.compare(expected).empty)
        self.assertTrue(output_.keys() == output.keys())

    def test_remove_redundant_index_levels_wo_opt_indices(self):
        output, _ = get_sim_output_data_frame(redundant_idx=True)
        output_ = UQExperiment.remove_redundant_index_levels(output)

        expected, _ = get_sim_output_data_frame(redundant_idx=False)

        self.assertTrue(output_.compare(expected).empty)
        self.assertTrue(output_.keys() == output.keys())

    def test_post_process_data_average_repetitions(self):

        n_runs = 10
        output1, _ = get_sim_output_data_frame(n_runs=n_runs, n_peds=2, n_tsteps=2)
        output2, _ = get_sim_output_data_frame(n_runs=n_runs, n_peds=2)
        output3, _ = get_sim_output_data_frame(n_runs=n_runs)
        sim_output = {0: output1, 1: output2, 2: output3}
        sim_output_sizes = [sim_output[i].size for i in range(3)]

        self.dummy_experiment.sim_output = sim_output
        self.dummy_experiment.average_repetitions_sim_output()
        UQExperiment.apply_functions_to_qoi(self.dummy_experiment, post_process_apps=[])
        sim_output__sizes_times_n_runs = [
            self.dummy_experiment.sim_output[i].size * n_runs for i in range(3)
        ]

        idx_names = [self.dummy_experiment.sim_output[i].index.names for i in range(3)]

        self.assertEqual(sim_output__sizes_times_n_runs, sim_output_sizes)
        self.assertTrue("run_id" not in idx_names)

    def test_average_model_output_repetitions_time_key(self):
        df, expected = get_sim_output_data_frame(n_tsteps=10)
        df_mean = UQExperiment.average_model_output_repetitions(df)
        self.assertTrue(np.equal(df_mean.to_numpy().transpose()[0], expected).all())

    def test_average_model_output_repetitions_ped_key(self):
        df, expected = get_sim_output_data_frame(n_peds=3)
        df_mean = UQExperiment.average_model_output_repetitions(df)
        self.assertTrue(np.equal(df_mean.to_numpy().transpose()[0], expected).all())

    def test_average_model_output_repetitions_ped_and_time_key(self):
        df, expected = get_sim_output_data_frame(n_peds=3, n_tsteps=10)
        df_mean = UQExperiment.average_model_output_repetitions(df)
        self.assertTrue(np.equal(df_mean.to_numpy().transpose()[0], expected).all())

    def test_average_model_output_repetitions_no_key(self):
        df, expected = get_sim_output_data_frame()
        df_mean = UQExperiment.average_model_output_repetitions(df)
        self.assertTrue(np.equal(df_mean.to_numpy().transpose()[0], expected).all())


if __name__ == "__main__":
    unittest.main()
