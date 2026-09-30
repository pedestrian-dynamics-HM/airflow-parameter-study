import operator
import os.path
import pickle as pkl
import shutil
import warnings
from datetime import datetime
from enum import Enum
from typing import List

import numpy as np
import pandas as pd

from vimuq import _get_experiment_pkl_filename
from vimuq.uq.model import Model, VadereModel
from vimuq.uq.uq_method import UQMethod
from vimuq.util.post_process_application import PostProcessApplication
from vimuq.util.git import get_git_revision_hash


class Stage(Enum):
    PRE_RUN = 1
    POST_RUN = 2
    POST_ANALYSIS = 3


class UQExperiment:
    SUBDIR = "summary"
    SCENARIO_SUBDIR = "scenario"
    TSTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

    def __init__(
        self,
        model: Model,
        method: UQMethod,
        path2output: str,
        parameter_variations=None,
    ):
        self.model = model
        self.method = method
        self.parameter_variations = parameter_variations
        self.path2output = path2output
        self.meta_info = dict()
        self.sim_output = None
        self.uq_output = None
        self.stage = Stage.PRE_RUN
        self.average_repetitions = ""
        self.analysis_n_samples = None

    def run(self):
        """Runs the uncertainty quantification experiment, that is it generates
        samples and evaluates the model. Logs the experiment and saves the
        output to path2output."""

        start = datetime.now()
        eval_dict = dict()

        os.makedirs(self.path2output, exist_ok=False)

        # Copy scenario file to output folder. The copied version in the
        # experiment output folder should not be altered. It simplifies
        # re-running the experiment at a later point in time with the original
        # scenario file, while the scenarios in the model folder may have been
        # changed (e.g. migrated).
        if isinstance(self.model, VadereModel):
            dest = os.path.join(self.path2output, self.SCENARIO_SUBDIR)
            os.makedirs(dest, exist_ok=False)
            if os.path.isdir(self.model.path2scenario):
                shutil.copytree(self.model.path2scenario, dest, dirs_exist_ok=True)
            else:
                shutil.copy(self.model.path2scenario, dest)

        # generate samples - only if they're None, otherwise use existing variations
        # TODO and if repeat_variations is set
        if self.parameter_variations is None:
            self.parameter_variations = self.method.sample(self.model.parameter_space)
        # evaluate samples
        eval_dict["meta_info"], self.sim_output = self.model.evaluate(
            self.parameter_variations, self.path2output
        )

        self.stage = Stage.POST_RUN

        stop = datetime.now()
        eval_dict["start"] = start.strftime(self.TSTAMP_FORMAT)
        eval_dict["stop"] = stop.strftime(self.TSTAMP_FORMAT)
        eval_dict["run_time"] = (stop - start).total_seconds()
        eval_dict["commit_hash"] = get_git_revision_hash()
        self.meta_info["evaluation"] = eval_dict

    def pre_analysis_clean_up(self):
        # remove sim output file extensions
        self.sim_output = {
            key.split(".")[0]: self.sim_output[key] for key in self.sim_output.keys()
        }

    def check_simulation_output(self, control_qoi, end_times, service_times):
        meta_key = "pre_analysis"
        group_by = ["id", "run_id"]

        end_times_key = "end_times"
        end_times_checked = False
        service_times_key = "service_times"
        service_times_checked = False

        if meta_key in self.meta_info.keys():
            pre_analysis_keys = self.meta_info["pre_analysis"].keys()
            if end_times_key in pre_analysis_keys:
                end_times_checked = True
            if service_times_key in pre_analysis_keys:
                service_times_checked = True

        if end_times and not end_times_checked:
            invalid_jobs = self.check_end_times_ok(control_qoi, group_by)
            self.meta_info[meta_key] = self.get_check_meta_info(
                check_name=end_times_key,
                success=not invalid_jobs,
                invalid_jobs=invalid_jobs,
            )

        if service_times and not service_times_checked:
            invalid_jobs = self.check_service_times_ok(
                control_qoi, group_by, service_times
            )
            self.meta_info[meta_key] = self.get_check_meta_info(
                check_name=service_times_key,
                success=not invalid_jobs,
                invalid_jobs=invalid_jobs,
            )

    def check_service_times_ok(self, control_qoi, group_by, service_times):
        """Checks for each simulation if pedestrians reached the target, i.e.
        end time is not infinity. Should only be applied to queue scenarios
        with constant waiting times.

        Arguments
        ----------
        control_qoi :   str; Simulation output that is used for checking;
                        Typically, this is the pedIdKey_pedEndTime.txt.
        service_times : List [mean_service_time, +-rel_deviation]
        group_by :      List of str (indices) of the dataframe

        Returns
        ----------
        invalid_jobs :   List of tuples [(id, run_id), ...]
        """

        allowed_min = service_times[0] * (1 - service_times[1])
        allowed_max = service_times[0] * (1 + service_times[1])
        min_exceeded = list()
        max_exceeded = list()
        jobs = self.sim_output[control_qoi].groupby(group_by)
        for key, data in jobs:
            data_ = data.sort_values(data.columns[0]).diff().copy()
            min_value = data_.min(axis=0, skipna=True)[0]
            max_value = data_.max(axis=0, skipna=True)[0]

            if min_value < allowed_min:
                min_exceeded.append(key)
            if max_value > allowed_max:
                max_exceeded.append(key)
        msg = str()

        invalid_jobs = min_exceeded + max_exceeded
        if invalid_jobs:
            if min_exceeded:
                msg += f"{min_exceeded.__len__()}/{jobs.__len__()} simulations (id, run_id) < min: {str(min_exceeded)}\n"
            if max_exceeded:
                msg += f"{max_exceeded.__len__()}/{jobs.__len__()} simulations (id, run_id) > max: {str(max_exceeded)}\n"
            warnings.warn(msg)
        else:
            msg = "Service times within legal limits. Simulations ran successfully."
            print(msg)  # ToDo [Logger] turn print(...) into logger info
        return invalid_jobs

    def check_end_times_ok(self, control_qoi, group_by):
        """Checks for each simulation if pedestrians reached the target, i.e.
        end time is not infinity.

        Returns
        ----------
        invalid_jobs :   List of tuples [(id, run_id), ...]
        """

        invalid_jobs = list()

        jobs = self.sim_output[control_qoi].groupby(group_by)
        for key, data in jobs:
            if np.isinf(data.to_numpy().T[0]).any():
                invalid_jobs.append(key)
        if invalid_jobs:
            msg = f"{invalid_jobs.__len__()}/{jobs.__len__()} simulations (id, run_id) failed due to pedestrian end time == inf: {invalid_jobs}"
            warnings.warn(msg)
        else:
            msg = "All pedEndTimes ok / pedestrians reached their target. Simulations ran successfully."
            print(msg)  # ToDo [Logger] turn print(...) into logger info
        return invalid_jobs

    @staticmethod
    def get_check_meta_info(check_name: str, success: bool, invalid_jobs: List):
        return {check_name: dict(success=success, invalid_jobs=invalid_jobs)}

    def analyze(
        self,
        qois: List[str] = list(),
        qoi_app_functions: List[PostProcessApplication] = list(),
        average_repetitions: str = "",
        n_samples: int = None,
    ):
        """Processes the simulation data, i.e. cleans the data, averages data
        from repeated simulations, calculates sensitivity indices and
        statistical measures.

        Arguments
        ----------
        qois :                  Quantities of interest that will be analyzed.
        average_repetitions :   If simulations have been run multiple times for
                                for the same sample, average the model output
                                over all repetitions.
        qoi_app_functions :     List of PostProcessApplications
        n_samples :             Number of samples that are considered when
                                calculating the sensitivity by means of the
                                Sobol' sequence.

        """

        start = datetime.now()
        analysis_dict = dict()

        self.clean_sim_output()

        self.average_repetitions = average_repetitions
        if average_repetitions == "A":
            self.average_repetitions_sim_output()

        # post-process simulation data
        self.apply_functions_to_qoi(qoi_app_functions)

        if average_repetitions == "B":
            self.average_repetitions_sim_output()

        # compute sensitivity indices
        # if no QoI is given, analyze all possible simulation outputs
        if not qois:
            qois = self.sim_output.keys()
        if n_samples:
            self.analysis_n_samples = n_samples
        self.uq_output = self.method.analyze(
            self.model.parameter_space, self.sim_output, qois, n_samples
        )

        if average_repetitions == "C":
            self.average_repetitions_uq_output()

        # update stage
        self.stage = Stage.POST_ANALYSIS

        # log some meta info
        stop = datetime.now()
        analysis_dict["start"] = start.strftime(self.TSTAMP_FORMAT)
        analysis_dict["stop"] = stop.strftime(self.TSTAMP_FORMAT)
        analysis_dict["run_time"] = (stop - start).total_seconds()
        analysis_dict["average_repetitions"] = average_repetitions
        analysis_dict["commit_hash"] = get_git_revision_hash()
        analysis_dict["n_samples"] = self.analysis_n_samples
        self.meta_info["analysis"] = analysis_dict

    def clean_sim_output(self):
        """ "Removes redundant index levels from any QoI."""

        for qoi, df in self.sim_output.items():
            df = self.remove_redundant_index_levels(df)

            self.sim_output[qoi] = df

    def average_repetitions_sim_output(self):
        """Averages simulation results over all runs (repetitions) for each
        QoI."""

        for qoi, df in self.sim_output.items():
            df = self.average_model_output_repetitions(df.copy())

            self.sim_output[qoi] = df.copy()

    def average_repetitions_uq_output(self):
        """Averages UQ results over all runs (repetitions) for each QoI."""

        for qoi, df in self.uq_output.items():
            if "param" in df.columns:
                # this is the case for sensitivity indices
                df = self.average_sensitivity_indices(df.copy())
            else:
                # this is the case for forward propagation results
                df = self.average_forward_propagation_results(df.copy())

            self.uq_output[qoi] = df.copy()

    @staticmethod
    def average_forward_propagation_results(data_frame: pd.DataFrame):
        dummy_index = "dummy_index"
        group_idx = data_frame.index.names.copy()
        if len(group_idx) == 2 and "run_id" in group_idx:
            remaining_index = group_idx.copy()
            remaining_index.remove("run_id")
            remaining_index = remaining_index[0]
        elif group_idx == ["run_id"]:
            # insert dummy index
            remaining_index = dummy_index
            group_idx.append(remaining_index)
            data_frame.insert(
                loc=0, column=remaining_index, value=np.arange(data_frame.shape[0])
            )
            data_frame.set_index(keys=remaining_index, append=True, inplace=True)
        else:
            msg = "I don't know what to calculate the average of."
            raise UserWarning(msg)

        # TODO replace print by logger
        msg = (
            "I'm taking the average of all UQ results for each "
            f"repetition. Remaining index: {remaining_index}"
        )
        print(msg)

        group_idx.remove("run_id")  # because we want to average over all runs
        data_frame = data_frame.groupby(group_idx).mean()

        if dummy_index in data_frame.index.names:
            data_frame.reset_index(level=dummy_index, drop=True, inplace=True)
        return data_frame

    @staticmethod
    def average_sensitivity_indices(data_frame: pd.DataFrame):
        params = data_frame["param"].drop_duplicates().to_list()
        data_frame.insert(loc=0, column="sort_idx", value=np.nan)
        for p in range(len(params)):
            data_frame.loc[data_frame["param"] == params[p], "sort_idx"] = p
        df = data_frame.set_index(["param", "sort_idx"], append=True)

        group_idx = df.index.names.copy()
        group_idx.remove("run_id")  # because we want to average over all runs
        df = df.groupby(group_idx).mean()
        df.reset_index(level="param", inplace=True)
        group_idx.remove("param")
        group_idx.remove(
            "sort_idx"
        )  # remove and add sort_idx to assure that it is at the last position
        group_idx.append("sort_idx")
        df.sort_index(level=group_idx, inplace=True)
        df.reset_index(level="sort_idx", drop=True, inplace=True)
        return df

    def apply_functions_to_qoi(self, post_process_apps: List[PostProcessApplication]):
        """Post-processes data frames for each quantity of interest. The base
        output is not altered, but mathematical functions are applied to the
        base QoI and the result is stored as additional QoI. For example, let
        base QoI: 'qoi_1';
        functions to be applied: 'min', 'max';
        resulting QoIs: 'qoi_1', 'qoi_1_min', 'qoi_1_max'

        Arguments
        ----------
        post_process_apps : list of mathematical functions to be applied to
                            the simulation output.
        """

        # use temp dict because len of sim_output must not change within loop
        temp_sim_output = dict()

        for qoi, df in self.sim_output.items():
            for post_app in post_process_apps:
                if qoi == post_app.qoi:
                    new_qoi, new_df = self.apply_function(df, post_app)
                    temp_sim_output[new_qoi] = new_df

        self.sim_output.update(temp_sim_output)

    @staticmethod
    def remove_redundant_index_levels(data_frame):
        """Removes redundant index levels from simulation output.
        In case of a scalar qoi, the dataframe may include a redundant index
        level (df.index.name = None) with the same index value for each row.
        This disturbs processing the data. Therefore, drop these indices.
        """

        df_idx_names = data_frame.index.names.copy()
        while None in df_idx_names:
            df_none_level = df_idx_names.index(None)
            data_frame = data_frame.droplevel(level=df_none_level)
            df_idx_names = data_frame.index.names.copy()

        return data_frame

    @staticmethod
    def average_model_output_repetitions(data_frame: pd.DataFrame):
        """Averages model output over all repetitions (column 'run_id' in
        data_frame)."""

        index_to_average = "run_id"

        df_indices = data_frame.index.names.copy()
        if index_to_average in df_indices:
            df_indices.remove(
                index_to_average
            )  # get list of indices that are not 'run_id', e.g. ['id','pedestrianId']
            data_frame = data_frame.groupby(df_indices).aggregate(func="mean")
        else:
            # ToDo [Logger] turn print(...) into logger info
            print("No repetitions found. The simulation output was not averaged.")

        return data_frame

    @staticmethod
    def apply_function(data_frame: pd.DataFrame, post_app: PostProcessApplication):
        """Applies a function to data frame, which is grouped by a given index.

        Arguments
        ----------
        data_frame :    DataFrame to be processed
        post_app :      PostProcessApplication

        Returns
        ----------
        qoi_ :          str, concatenated old qoi name and name of the function
                        that has been applied
        data_frame_ :   new DataFrame
        """

        df = data_frame.copy()

        valid_indices = df.index.names
        for idx in post_app.indices:
            if idx not in valid_indices:
                raise ValueError(
                    f"Argument group_by_index must not contain {idx}. Valid "
                    f"index names are {valid_indices}"
                )

        if "run_id" in valid_indices:
            post_app.add_index_run_id()

        # Replace all values that do not match condition such that skipna=True
        # applies when .agg('xy'). The condition is:
        # <value from the dataframe> <post_app.operator> <post_app.threshold>
        # Example: post_app.operator = ">", post_app.threshold = 0
        # then all values in the dataframe (column 0) <= 0 will be replaced by
        # pd.NA, i.e. all other values that match the condition are considered
        # when the function is applied
        rel_ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }
        if (post_app.operator is not None) and (post_app.threshold is not None):
            df[~rel_ops[post_app.operator](df.iloc[:, 0], post_app.threshold)] = pd.NA

        qoi_ = post_app.get_new_qoi_name()

        # This is not necessary as long as the output processor yields only
        # one output
        if df.columns.size > 1:
            raise UserWarning("I don't know which column I should choose.")
        else:
            col = df.columns.to_list()[0]

        data_frame_ = df.groupby(post_app.indices)
        # This is a workaround because DataFrameGroupBy.sum() and .agg("sum")
        # did/does not work. Therefore, we use apply function instead. We use it
        # only for summation because DataFrameGroupBy.agg is much faster.
        # If necessary, the apply function could be used for any user defined
        # function
        # --- workaround start ---
        if post_app.function == "sum":
            data_frame_ = data_frame_.apply(lambda x: x.sum())
        elif post_app.function == "rank":
            # replace column pedestrianId by rank of E and sort rank in
            # ascending order
            old_index = "pedestrianId"
            new_index = post_app.function
            if old_index not in df.index.names:
                raise UserWarning(
                    f"Cannot replace index {old_index}"
                    f"because it is not present in dataframe."
                )
            else:
                df[new_index] = df[col]
                data_frame_ = df.groupby(post_app.indices)
                data_frame_ = data_frame_[new_index].agg(
                    func="rank", method="first", ascending=False
                )
                data_frame_ = pd.concat([data_frame.copy(), data_frame_], axis=1)
                data_frame_.reset_index(level=old_index, drop=True, inplace=True)
                data_frame_.set_index(keys=new_index, append=True, inplace=True)
                data_frame_.sort_index(inplace=True)
        else:
            data_frame_ = data_frame_.agg(post_app.function)
        # --- workaround end ---

        return qoi_, data_frame_

    def save(self):
        """Saves an experiment using pickle. Duplicates get an identifier
        (YYYY-MM-DD_hh-mm-ss). The suffix of the filename is generated
        automatically."""

        path = os.path.join(self.path2output, self.SUBDIR)
        os.makedirs(path, exist_ok=True)

        path2file = os.path.join(
            path,
            _get_experiment_pkl_filename(
                stage=self.stage,
                avrg_technique=self.average_repetitions,
                n_samples=self.analysis_n_samples,
            ),
        )

        if os.path.exists(path2file):
            suffix = datetime.now().strftime(self.TSTAMP_FORMAT)
            new_filename = _get_experiment_pkl_filename(
                stage=self.stage,
                avrg_technique=self.average_repetitions,
                n_samples=self.analysis_n_samples,
                suffix=suffix,
            )
            path2file = os.path.join(path, new_filename)

        with open(path2file, "wb") as outp:
            pkl.dump(self, outp, pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def read_experiment(
        path_to_experiment_dir: str,
        stage: Stage,
        avrg_technique: str = "",
        n_samples: str = None,
        suffix: str = "",
    ):
        """Loads the stage of an experiment from existing files.

        Arguments
        ----------
        path_to_experiment_dir :    Absolute path to directory in which the
                                    files 'uq_experiment*.pkl' are stored.
        stage :                     Stage (version of the experiment) that
                                    will be loaded.
        avrg_technique :            Technique that was used to process model
                                    outputs for several repetitions
        n_samples :                 Number of samples that were considered when
                                    calculating sensitivity indices (analysis).
        suffix :                    Specifies filename if multiple files
                                    with same prefix and stage exist.
                                    Default: None

        Returns
        ----------
        experiment :                UQExperiment

        """

        path_to_experiment = os.path.join(
            path_to_experiment_dir,
            UQExperiment.SUBDIR,
            _get_experiment_pkl_filename(
                stage=stage,
                avrg_technique=avrg_technique,
                suffix=suffix,
                n_samples=n_samples,
            ),
        )
        if os.path.exists(path_to_experiment):
            # with open(path_to_experiment, "rb") as inp:
            experiment = pkl.load(open(path_to_experiment, "rb"))
        else:
            raise ValueError("Path {} is invalid".format(path_to_experiment))

        experiment = UQExperiment._update_experiment(experiment)

        return experiment

    @staticmethod
    def _update_experiment(experiment):
        experiment_ = UQExperiment(
            model=experiment.model,
            method=experiment.method,
            path2output=experiment.path2output,
        )

        for k, v in experiment.__dict__.items():
            experiment_.__setattr__(k, v)

        return experiment_
