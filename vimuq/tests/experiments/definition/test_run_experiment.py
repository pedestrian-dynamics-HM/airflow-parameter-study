import unittest
import os

from vimuq.experiments.definition import run_experiment
from vimuq.tests import remove_output_dirs, _get_head_path
from vimuq.tests.imports import v_experiment, pm_experiment
from vimuq.uq.distribution import Distribution
from vimuq.uq.model import Model, PyModel
from vimuq.uq.parameter_space import ParameterSpace
from vimuq.uq.uq_method import UQMethod
from vimuq.util.argv import split_argument_list, _str_to_bool
from vimuq.util.postchanges_helper import ChangeInitialPathogenLoad


class TestRunExperimentParser(unittest.TestCase):
    """Test class for testing vimuq.experiments.definition.run_experiment."""

    def setUp(self):
        self._parser = run_experiment._get_parser()

    def test__parameter_dist_type_check_throws_error(self):
        argv_ = ["--parameter", "paramX", "ANotSupportedDistribution", "1", "2"]
        args = self._parser.parse_args(argv_)
        self.assertRaises(
            ValueError, run_experiment._parameter_dist_type_check, args.parameter
        )

    def test__parameter_dist_type_check_ok(self):
        """Checks that all implemented subclasses of
        vimuq.util.distribution.Distribution can actually be parsed"""

        argv_ = []
        expected_dists = []
        i = 1
        for cls in Distribution.__subclasses__():
            expected_dists.append(cls.__name__)
            argv_ += ["--parameter", f"parameter{i}", cls.__name__, "0", "2"]
            i += 1
        args = self._parser.parse_args(argv_)

        dists = []
        for p in args.parameter:
            dists.append(p[1])

        self.assertEqual(dists, expected_dists)

    def test__parameter_dist_metrics_str_to_float(self):
        args = self._parser.parse_args(v_experiment["argv_f"])
        params = args.parameter
        run_experiment._parameter_dist_metrics_str_to_float(params)

        check_sum = 0
        for p in params:
            check_sum += sum(p[2:4])

        self.assertEqual(check_sum, 13.5)

    def test__get_parser(self):
        args = self._parser.parse_args(v_experiment["argv_f"])
        expected_args = v_experiment["namespace"]

        self.assertEqual(args, expected_args)

    def test__get_arguments(self):
        args = run_experiment._get_arguments(v_experiment["argv_f"])
        self.assertEqual(args, v_experiment["args"])

    def test__str_to_bool_ok(self):
        expected = [True, False]
        tests = [str(x) for x in expected]
        args = []
        for t in tests:
            args += [_str_to_bool(t)]
        self.assertEqual(args, expected)

    def test__str_to_bool_error(self):
        bad_str = "other_than_True_or_False"
        self.assertRaises(ValueError, _str_to_bool, bad_str)


class TestExperimentSetup(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        path2project = _get_head_path(cwd, "airflow-uq")
        path2tests = os.path.join(path2project, "vimuq", "tests")

        self._path = "."
        self._id = "TEST_ID"

        self._model = Model(
            ParameterSpace(),
            list(),
            os.path.join(
                path2tests, "model", "vadere", "simulator", "test_vadere-console.txt"
            ),
        )
        self._experiment = run_experiment._setup_uq_experiment(
            self._path, self._id, UQMethod(), self._model
        )

    def tearDown(self):
        remove_output_dirs(self._path, self._id)

    def test__setup_uq_experiment_no_duplicate(self):
        """Checks target dir for new experiment."""

        target_dir = os.path.join(self._path, self._id)
        self.assertEqual(self._experiment.path2output, target_dir)

    def test__setup_uq_experiment_duplicate(self):
        """Checks target dir for new experiment that already exists at least
        once."""

        # this represents running an experiment and saving it to the target dir
        os.makedirs(self._experiment.path2output, exist_ok=False)

        experiment_duplicate = run_experiment._setup_uq_experiment(
            self._path, self._id, UQMethod(), self._model
        )
        self.assertNotEqual(
            self._experiment.path2output, experiment_duplicate.path2output
        )


class TestMethodSetup(unittest.TestCase):
    def test__setup_uq_method_throws_error(self):
        uq_method = "ClassThatIsCertainlyNotImplemented"
        dummy_int = 2
        self.assertRaises(
            AttributeError,
            run_experiment._setup_uq_method,
            uq_method,
            dummy_int,
        )

    def test__setup_uq_method_passes(self):
        method_names = [cls.__name__ for cls in UQMethod.__subclasses__()]
        dummy_int = 2

        for m in method_names:
            method = run_experiment._setup_uq_method(m, dummy_int)
            method_cls_name = method.__class__.__name__

            self.assertEqual(method_cls_name, m)


class TestModelSetup(unittest.TestCase):
    def test__setup_model_vadere_postchanges(self):
        args = v_experiment["args"]
        params = run_experiment._get_model_input(args)
        model = run_experiment._setup_model(args, params)
        post_changes = model.suqc_post_changes
        post_changes_key_set = set(
            post_changes.__dict__["_apply_scenario_changes"].keys()
        )

        self.assertEqual(
            post_changes_key_set,
            {
                "scenario_name",
                "real_time_sim_time_ratio",
                "always_enable_meta_data",
                "description",
                ChangeInitialPathogenLoad(1).name,
            },
        )


class TestIshigamiModel(unittest.TestCase):
    def setUp(self):
        argv = split_argument_list(pm_experiment["argv"])
        self.args = run_experiment._get_arguments(argv)

    def tearDown(self):
        remove_output_dirs(self.args.outputpath, self.args.experimentid)

    def test__setup_model_pymodel(self):
        model = run_experiment._setup_model(self.args, ParameterSpace())
        self.assertTrue(isinstance(model, PyModel))

    def test__run_experiment_with_ishigami(self):
        """Test function following the example in [1].

        Expected results for Sobol’ analysis with the Saltelli sampling and 1024
        samples: x1: 0.31683154 x2: 0.44376306 x3: 0.01220312

        References
        ----------
        [1] https://salib.readthedocs.io/en/latest/getting-started.html,
        accessed on 2022-07-21

        """

        experiment = run_experiment._setup(self.args)
        experiment.run()
        experiment.analyze()

        digits = 8
        self.assertAlmostEqual(
            first=experiment.uq_output[self.args.output[0]]["S1"][0],
            second=0.31683154,
            places=digits,
        )
        self.assertAlmostEqual(
            first=experiment.uq_output[self.args.output[0]]["S1"][1],
            second=0.44376306,
            places=digits,
        )
        self.assertAlmostEqual(
            first=experiment.uq_output[self.args.output[0]]["S1"][2],
            second=0.01220312,
            places=digits,
        )


if __name__ == "__main__":
    unittest.main()
