import errno
import os
import shutil
import subprocess
import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from suqc import (
    VadereEnvironmentManager,
    UserDefinedSampling,
    FolderExistScenarios,
    VadereScenarioCreation,
    QuantityOfInterest,
)
from suqc.CommandBuilder.JarCommand import JarCommand
from suqc.parameter.postchanges import ChangeRandomNumber, PostScenarioChangesBase
from suqc.request import DictVariation, VariationBase, Request
from importlib import import_module

from vimuq.uq.parameter_space import ParameterSpace


class Model:
    """Base class for models to be analyzed with uq methods"""

    OUTPUT_SUBDIR = "simulations"

    def __init__(
        self,
        parameter_space: ParameterSpace,
        quantities_of_interest: List[str],
        path2model: str,
    ):
        self.parameter_space = parameter_space
        self.quantities_of_interest = quantities_of_interest
        self.path2model = path2model

    @staticmethod
    def _check_path_exists(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), os.path.abspath(path)
            )
        else:
            return True

    def evaluate(self, input, path2output: str = os.path.join("..", "output")):
        """Evaluates the model for a given input. If the model generates an
        output itself and path2output is not None, save the simulation output to
        a subdir of path2output.

        Arguments
        ----------
        input :         variations for n parameters formatted as list of
                        dictionaries. Example:
                        [{'param_1': 1.0, ..., 'param_n': 100.0},
                        ...,
                        {'param_1': 10.0, ..., 'param_n': 1000.0}]

        path2output :   parent dir in which output is saved

        Returns
        ----------
        meta_info :     if present, pandas.dataframe containing information
                        about simulation runs

        model_output :  dictionary

        """
        pass


class VadereModel(Model):
    """Class for Vadere models"""

    # defines the generation of random numbers that are used by suq-controller
    # as simulation seeds in Vadere scenario files; ranges in [0, 2 ** 21 -1]
    RANDINT_SEED_VADERE = 23812

    def __init__(
        self,
        parameter_space: ParameterSpace,
        quantities_of_interest: List[str],
        path2model: str,
        path2scenario: str,
        run_local: bool = True,
        repetitions: int = 1,
        suqc_post_changes: Union[PostScenarioChangesBase, None] = None,
        use_scenario_files_as_repetitions: bool = False,
    ):
        super().__init__(parameter_space, quantities_of_interest, path2model)

        # TODO check if its a file or folder
        # if self._check_path_exists(path2scenario):
        self.path2scenario = path2scenario

        self.repetitions = repetitions
        self.repetitions_seeds = None
        self.use_scenario_files_as_repetitions = use_scenario_files_as_repetitions

        self.run_local = run_local

        # this cannot be passed as default argument, otherwise it may happen
        # that suqc_post_changes is not instantiated correctly
        if suqc_post_changes is None:
            self.suqc_post_changes = PostScenarioChangesBase(apply_default=True)
        else:
            self.suqc_post_changes = suqc_post_changes

        # The random generator must be instantiated with every instance of
        # VaderModel
        self.rng = np.random.default_rng(seed=self.RANDINT_SEED_VADERE)

        self.path2model = self._get_vadere_model()

    def _get_vadere_model(self):
        """Checks if path to user-defined vadere-console.jar exists. This jar
        file can be provided by the user. If this model does not exist, uses
        vadere-console.jar from the vadere submodule. If this does not exist
        yet, creates vadere-console.jar using maven."""

        if not os.path.exists(self.path2model):
            warnings.warn(
                f"{self.path2model} does not exist. Vadere is built from submodule."
            )
            absPath2vadere = self._get_abs_path_to_vadere_submodule()
            self._build_vadere_console_jar(absPath2vadere)

            scr_path = os.path.join(
                absPath2vadere, "VadereSimulator", "target", "vadere-console.jar"
            )

            # copy jar file; replace jar file if it already exists
            head, _ = os.path.split(self.path2model)
            os.makedirs(head, exist_ok=True)
            dest_path = os.path.join(head, "vadere-console_submodel-build.jar")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.copy(scr_path, dest_path)

            return dest_path
        else:
            return self.path2model

    @staticmethod
    def _get_abs_path_to_vadere_submodule():
        cwd = os.getcwd()
        root_name = "airflow-uq"
        cwd_list = cwd.split(root_name)[:-1]
        absPath2root = root_name.join(cwd_list) + root_name
        absPath2vadere = os.path.join(absPath2root, "vadere")
        return absPath2vadere

    @staticmethod
    def _build_vadere_console_jar(path2vadere_root_dir):
        p = subprocess.Popen(
            ["mvn", "clean", "-Dmaven.test.skip=true", "package"],
            cwd=path2vadere_root_dir,
            shell=True,
        )
        p.wait()

    def _set_repetitions_seeds(self):
        # lowest to highest signed int32 [-2147483648 to 2147483647]
        self.repetitions_seeds = self.rng.integers(
            low=-2147483648, high=2147483647, size=self.repetitions
        ).tolist()

    def apply_suqc_rand_number_per_repetition(self):
        """
        Defines changes which the suq-controller applies to the original
        scenario file before the model is evaluated. The seed specifies the
        generation of random numbers. This should always be applied if results
        must be reproducible.
        """

        self._set_repetitions_seeds()

        change_rand_numbers = ChangeRandomNumber(fixed_per_run=True)
        change_rand_numbers.set_fixed_random_nr(self.repetitions_seeds)
        self.suqc_post_changes.add_scenario_change(change_rand_numbers)

    def run_suqc_scenario_folder(self, model_input, qoi, jar_command, path2output):
        """Runs the suq-controller setup for a scenario folder.

        Arguments
        ----------
        model_input : List of parameter variations
        qoi : List of quantities of interest
        jar_command : JarCommand object
        path2output : Path to output directory
        """
        njobs = 12

        parameter_variation = UserDefinedSampling(model_input)
        parameter_variation = parameter_variation.multiply_scenario_runs(
            scenario_runs=1
        )
        request_item_list = []

        scenario_files = sorted(
            [f for f in os.listdir(self.path2scenario) if f.endswith(".scenario")]
        )
        for idx, filename in enumerate(scenario_files):
            scenario_path = os.path.join(self.path2scenario, filename)
            env_man = VadereEnvironmentManager.create_variation_env(
                basis_scenario=scenario_path,
                base_path=path2output,
                env_name=self.OUTPUT_SUBDIR,
                handle_existing="write_in_if_exist_else_create",
            )
            parameter_variation.points.index = (
                parameter_variation.points.index.set_levels([idx], level="run_id")
            )
            scenario_creation = VadereScenarioCreation(env_man, parameter_variation)
            request_item_list.extend(
                scenario_creation.generate_scenarios(njobs, reset_folder=False)
            )

        request = Request(
            request_item_list=request_item_list,
            model=jar_command,
            qoi=QuantityOfInterest(qoi),
        )

        if self.run_local:
            model_output, meta_info = request.run(njobs)
        else:
            model_output, meta_info = request.remote(-1)
        if isinstance(model_output, pd.DataFrame):
            model_output = {self.quantities_of_interest[0]: model_output}
        return meta_info, model_output

    def run_suqc_scenario_file(self, model_input, qoi, jar_command, path2output):
        """Runs the suq-controller setup.

        Arguments
        ----------
        model_input : List of parameter variations
        qoi : List of quantities of interest
        jar_command : JarCommand object
        path2output : Path to output directory
        """

        suqc_setup = DictVariation(
            scenario_path=self.path2scenario,
            parameter_dict_list=model_input,
            qoi=qoi,
            model=jar_command,
            scenario_runs=self.repetitions,
            post_changes=self.suqc_post_changes,
            output_path=path2output,
            output_folder=self.OUTPUT_SUBDIR,
            remove_output=False,
        )

        if self.run_local:
            meta_info, model_output = suqc_setup.run(
                -1
            )  # -1 indicates to use all cores available to parallelize the scenarios
        else:
            meta_info, model_output = suqc_setup.remote(
                -1
            )  # -1 indicates to use all cores available to parallelize the scenarios

        if isinstance(model_output, pd.DataFrame):
            model_output = {self.quantities_of_interest[0]: model_output}
        return meta_info, model_output

    def evaluate(self, model_input, path2output=None):
        """Evaluates the Vadere model for varying inputs (m samples) by means of
        the suq-controller.

        Arguments
        ----------
        model_input :   variations for n parameters formatted as list of
                        dictionaries. Each element of the list is one sample,
                        that is len(model_input) = m. Example:
                        [{'param_1': 1.0, ..., 'param_n': 100.0},
                        ...,
                        {'param_1': 10.0, ..., 'param_n': 1000.0}]

        path2output :   parent dir in which output is saved

        Returns
        ----------
        meta_info :     pandas.dataframe
                        multi-index: id (of the parameter variation) / run_id
                        columns: Parameter/model_input[0], ...,
                        Parameter/model_input[m],
                        MetaInfo/required_wallclock_time, MetaInfo/return_code

        model_output :  dictionary
                        keys: Names of the Vadere output files, e.g.
                        'degOfExposurePerPed.txt'
                        values: pd.dataframe
                        multi-index: id (of the parameter variation) / run_id /
                        <vadere output specific indices, e.g. timeStep,
                        pedestrianId, ...>
                        columns: <output processor header>-PID<processor number>

        """
        jar_command = (
            JarCommand(jar_file=self.path2model)
            .add_option("-enableassertions")
            .main_class("suq")
        )

        qoi = [q + ".txt" for q in self.quantities_of_interest.copy()]

        if os.path.isfile(self.path2scenario):
            if os.path.splitext(self.path2scenario)[-1] == ".scenario":
                meta_info, model_output = self.run_suqc_scenario_file(
                    model_input, qoi, jar_command, path2output
                )
                return meta_info, model_output
        elif os.path.isdir(self.path2scenario):
            meta_info, model_output = self.run_suqc_scenario_folder(
                model_input, qoi, jar_command, path2output
            )
            return meta_info, model_output
        else:
            assert "path2scenario is invalid"
            return [], {}


class PyModel(Model):
    """A model based on the method 'model_function', which is defined in
    the python file located at 'path2model'. The actual model function takes an
    np.array as input and returns a 1D np.array, which contains scalar outputs
    for each sample. Non-scalar output quantities are not supported.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        quantities_of_interest: List[str],
        path2model: str,
    ):
        super().__init__(parameter_space, quantities_of_interest, path2model)

        self.model_function = getattr(
            import_module(self._convert_os_path_to_module(self.path2model)),
            "model_function",
        )

    @staticmethod
    def _convert_os_path_to_module(path):
        path = os.path.abspath(path)
        model, _ = os.path.basename(path).split(sep=".")  # remove file extension
        path = path.split(os.sep)[:-1]  # exclude filename from path
        vimuq_idx = np.max(np.where(np.array(path) == "vimuq"))
        module_list = path[vimuq_idx:] + [model]
        module_path = ".".join(module_list)

        return module_path

    def evaluate(self, input, path2output: str = None):
        """Evaluates a PyModel.

        Arguments
        ----------
        input :         variations for n parameters formatted as list of
                        dictionaries. Example:
                        [{'param_1': 1.0, ..., 'param_n': 100.0},
                        ...,
                        {'param_1': 10.0, ..., 'param_n': 1000.0}]

        path2output :   currently not required

        """
        array_input = pd.DataFrame(input).to_numpy()

        y = self.model_function(array_input)

        # make output compatible with output format in vimuq
        if len(y.shape) == 1:
            temp_df = pd.DataFrame(
                data=y,
                index=np.array(range(0, len(y))),
            )
            temp_df.index.name = "id"  # sample id
            temp_df.rename(columns={0: "Y"})  # model_output
            output = {self.quantities_of_interest[0]: temp_df}
        else:
            raise NotImplementedError(
                "PyModel output conversion for non-scalar output not yet implemented."
            )

        # ToDo if necessary: do something with the output, e.g. save to
        #  path2output

        meta_info = None
        return meta_info, output
