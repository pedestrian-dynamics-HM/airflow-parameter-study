from typing import List

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli

from vimuq.uq.distribution import UniformDistribution, Log10UniformDistribution
from vimuq.uq.parameter_space import ParameterSpace


class UQMethod:

    # defines the generation of random numbers that are used as samples for UQ
    # methods
    RANDINT_SEED_SAMPLING = 912523
    RANDOM = np.random.default_rng(RANDINT_SEED_SAMPLING)

    def __init__(self, sample_size: int = 0, calc_second_order: bool = True):
        self.sample_size: int = sample_size
        self.calc_second_order = calc_second_order

    def sample(self, parameter_space: ParameterSpace):
        pass

    @staticmethod
    def transform_samples(parameters, samples):
        samples_ = samples.copy()
        for i, p in enumerate(parameters):
            if isinstance(p.dist, Log10UniformDistribution):
                samples_[:, i] = p.dist.samples_log_to_lin_scale(samples[:, i])
        return samples_

    def analyze(
        self,
        parameter_space: ParameterSpace,
        model_output,
        qois: List[str],
        n_samples: int,
    ):
        """Analyzes the model output as defined by the chosen method

        Arguments
        ----------
        parameter_space :   Parameter space
        model_output :      Model output
        qois :              Quantities of interest that will be analyzed, all
                            other quantities of the simulation output will be
                            skipped.
        n_samples :         Number of samples to be considered. This should only
                            be applied in combination with the Sobol' sequence.
        """
        pass


class SALibSaltelliSobolSequence(UQMethod):
    def __init__(
        self,
        sample_size: int = 0,
        calc_second_order: bool = True,
        skip_samples: int = None,
    ):
        super().__init__(sample_size, calc_second_order)
        self.skip_samples = skip_samples

    def sample(self, parameter_space: ParameterSpace):
        """Generates Sobol sequence with SALib; Note that a seed for random
        number generator is not required"""

        salib_params = parameter_space.to_salib_format()

        # use default for skip_values, that is
        # max(2 ** ceil(log2(sample_size)), 16)
        salib_samples = saltelli.sample(
            salib_params,
            self.sample_size,
            calc_second_order=self.calc_second_order,
            skip_values=self.skip_samples,
        )

        parameter_names = [p.name for p in parameter_space.parameters]

        # transform samples of log uniform distribution
        salib_samples = UQMethod.transform_samples(
            parameters=parameter_space.parameters, samples=salib_samples
        )

        samples = array_to_par_var(salib_samples, parameter_names)

        return samples

    def analyze(
        self,
        parameter_space: ParameterSpace,
        model_output: dict,
        qois,
        n_samples,
    ):
        """Analyzes the model output with SALib Sobol method

        Arguments
        ----------
        see overridden method

        Returns
        ----------
        sensitivity_indices :   dictionary for each quantity of interest with
                                values: dataframe with multi-indices
                                run_id/pedestrianId/timeStep/parameter and
                                columns S1/S1_conf/ST/ST_conf/S2/S2_conf

        statistics :            dictionary for each quantity of interest with
                                values: dataframe with multi-indices
                                run_id/pedestrianId/timeStep and columns
                                mean/std/min/max/conf_interval

        """

        LOG_LEVEL = None  # for debugging or testing

        salib_params = parameter_space.to_salib_format()

        sensitivity_indices = dict()

        for qoi, data in model_output.items():
            if qoi in qois:
                # ToDo [Logger] turn print(...) into logger info
                print(f"Computing Sobol' indices for quantity of interest '{qoi}'.")
                si, df_indices = SALibSaltelliSobolSequence.compute(
                    salib_params,
                    data,
                    log_level=LOG_LEVEL,
                    n_samples=n_samples,
                    calc_second_order=self.calc_second_order,
                )
                if len(df_indices) > 0:
                    si = si.set_index(df_indices)
                sensitivity_indices[qoi] = si
            else:
                # ToDo [Logger] turn print(...) into logger info
                print(
                    f"Skipping computation of Sobol' indices for quantity of interest '{qoi}'."
                )

        return sensitivity_indices

    @staticmethod
    def compute(
        salib_params,
        df: pd.DataFrame,
        log_level=None,
        n_samples=None,
        calc_second_order=None,
    ):
        """Nested function calculates Sobol indices with SALib iteratively

        Arguments
        ----------
        log_level :     Takes either int or list of str and defines the level
                        to which the calculation process gets logged to console.
                        Integer defines the lowest logging level when processing
                        the data.
                        List[str], e.g. ['timeStep', 'pedestrianId'], defines
                        which index levels from df are taken into account.
                        Printing to console can
                        Default: None, no logging to console
        n_samples :     Number of samples to be considered.

        Returns
        ----------

        """

        df_indices = df.index.names.copy()

        # we do not want to iterate over the outputs that correspond to a sample
        # (parameter variation), which is indicated by index 'id'
        df_indices.remove("id")

        # only for logging / debugging purposes
        if isinstance(log_level, int):
            log_level = df_indices[: log_level + 1]

        if len(df_indices) == 0:

            # assure that output has the same order as the samples. In fact,
            # the dataframe indices should be in ascending order already.
            df = df.sort_index(ascending=True)

            salib_output = df.values[:, 0]  # convert shape (x,1) to (x,)

            # Take only the first N model outputs (samples). This is useful if
            # one wants to compare the sensitivity indices calculated with
            # different large sample sizes but one does not want to repeat the
            # model evaluations. N should be a power of two (2 ** n), with
            # n = 1, 2, 3, ...
            # If only first order SI is calculated, we need 2 ** n * (D + 2)
            # samples.
            # If second order SI is calculated, we need 2 ** n * (2 * D + 2)
            # samples.
            if n_samples:
                if calc_second_order:
                    N = n_samples * (2 * salib_params["num_vars"] + 2)
                elif not calc_second_order:
                    N = n_samples * (salib_params["num_vars"] + 2)
                else:
                    raise UserWarning(
                        "Cannot calculate correct number of"
                        "samples needed for the Sobol' analysis."
                    )
                salib_output = salib_output[:N]
            res = sobol.analyze(
                salib_params, salib_output, calc_second_order=calc_second_order
            )

            # convert Sobol indices to pandas dataframe
            temp = dict()
            temp["param"] = res.problem["names"]
            for k, v in res.items():
                temp[k] = list(v)
            si = pd.DataFrame(temp)

            si = SALibSaltelliSobolSequence.reformat_second_order_indices_dataframe(si)

        else:
            # Break down dataframe into slices. This iterative process is
            # necessary, since SALib's analysis methods cannot deal with
            # dataframes but require numpy arrays:

            # first index over which we iterate
            idx_name = df_indices[0]
            current_level = idx_name

            # convert index to column
            dff = df.reset_index(idx_name)

            # get all values from that column as list without duplicates
            idx_values = dff[idx_name][~dff[idx_name].duplicated()].to_list()

            # for each element in list, we obtain a slice that can be analyzed
            # separately
            si = None
            for idx_val in idx_values:

                # print level of process to console
                if isinstance(log_level, list):
                    if current_level in log_level:
                        print(
                            f"{current_level}: {str(idx_val)} / {str(max(idx_values))}"
                        )

                dff_slice = dff.loc[dff[idx_name] == idx_val]
                dff_slice = dff_slice.drop(columns=idx_name)
                si_, _ = SALibSaltelliSobolSequence.compute(
                    salib_params,
                    dff_slice,
                    log_level=log_level,
                    n_samples=n_samples,
                    calc_second_order=calc_second_order,
                )
                si_[idx_name] = idx_val
                if si is None:
                    si = si_
                else:
                    si = pd.concat([si, si_], ignore_index=True)

        return si, df_indices

    @staticmethod
    def to_salib_model_output(model_output):
        """Formats model output to structures that are compatible with SALib.

        Arguments
        ----------
        model_output :          pandas dataframe containing the model
                                evaluations for a single quantity of interest

        Returns
        ----------
        salib_model_output :    ndarray

        """
        model_out_array = model_output.to_numpy()  # shape (x,1)
        salib_model_output = model_out_array[:, 0]  # shape (x,)
        return salib_model_output

    @staticmethod
    def reformat_second_order_indices_dataframe(data_frame: pd.DataFrame):
        """If second order Sobol' indices are included in the analysis, we
        reformat the dataframe because we cannot easily deal with the structure
        returned by SALib. If second order indices were not computed, skip this.
        """

        skip_col = "param"  # the column does not contain uq results but only
        # the corresponding parameters; therefore, this column is excluded from
        # reformatting

        data_frame_ = pd.DataFrame(data_frame[skip_col].copy())
        data_frame_.set_index([skip_col], inplace=True)

        cols = data_frame.columns.to_list()
        cols.remove(skip_col)
        for col in cols:
            df_slice = data_frame[[skip_col, col]].copy()

            column_values = df_slice[col].values

            # convert all columns that contain an array in each cell instead of
            # scalar value; exclude str because str also has attribute __len__
            cell_is_array = [
                hasattr(cell, "__len__") and (not isinstance(cell, str))
                for cell in column_values
            ]
            if all(cell_is_array):
                params = df_slice[skip_col].drop_duplicates().to_list()

                df_ = pd.DataFrame
                for row in df_slice.iterrows():
                    p = row[1][0]
                    values = row[1][1]

                    new_cols = [(col, params[i]) for i in range(len(params))]
                    col_names = [skip_col] + new_cols
                    data = np.array([np.concatenate([np.array([p]), values])])
                    df_temp = pd.DataFrame(data, columns=col_names)
                    if df_.empty:
                        df_ = df_temp.copy()
                    else:
                        df_ = pd.concat([df_, df_temp.copy()])
                df_.set_index([skip_col], inplace=True)

                df_ = df_.astype(float)

                data_frame_ = pd.concat([data_frame_, df_], copy=True, axis=1)

            elif False in cell_is_array and True in cell_is_array:
                raise UserWarning(
                    f"Cannot convert column {col} because it contains both "
                    f"arrays and scalar values in different cells."
                )

            else:  # all cells contain scalar values -> no need to do anything
                df_slice.set_index([skip_col], inplace=True)
                data_frame_ = pd.concat([data_frame_, df_slice], copy=True, axis=1)

        data_frame_.reset_index(
            skip_col, inplace=True
        )  # we want to get the original index
        return data_frame_


class ChaosPyQuasiRandomMonteCarlo(UQMethod):
    """UQ Method for Monte Carlo simulations. Uses ChaosPy for quasi-random
    sampling.
    ToDo: add super class for forward propagation (and sensitivity analysis)
     techniques
    """

    def sample(self, parameter_space: ParameterSpace):

        j_dist = parameter_space.to_chaospy_joint_distribution()
        random_samples = j_dist.sample(
            size=self.sample_size,
            rule="random",
            seed=self.RANDINT_SEED_SAMPLING,
        )

        random_samples = random_samples.transpose()

        parameter_names = [p.name for p in parameter_space.parameters]

        # transform samples of log uniform distribution
        random_samples = UQMethod.transform_samples(
            parameters=parameter_space.parameters, samples=random_samples
        )

        random_samples = array_to_par_var(random_samples, parameter_names)

        return random_samples

    def analyze(
        self,
        parameter_space: ParameterSpace,
        model_output,
        qois: List[str],
        n_samples: int,
    ):

        statistics = dict()

        for qoi, data in model_output.items():
            if qoi in qois:
                # ToDo [Logger] turn print(...) into logger info
                print(
                    f"Computing summary statistics for quantity of interest "
                    f"'{qoi}'."
                )
                # if len(data.columns) > 1:
                #     raise UserWarning("Cannot treat more than one column ")
                # drop any index that is not "id"
                keep_idx = "id"
                groupby_idx = list(data.index.names)
                groupby_idx.remove(keep_idx)
                data_ = data.reset_index(level=groupby_idx)

                if len(groupby_idx) > 0:
                    data_ = data_.groupby(groupby_idx)
                temp_stats = data_.describe(
                    percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                )

                # Reformat resulting dataframe to get the same structure as
                # the results from data_ as GroupBy
                # This is probably not the best way to do that.
                # ToDo refactoring
                if len(groupby_idx) == 0:
                    temp_stats = pd.DataFrame(
                        data=temp_stats.unstack().values,
                        index=pd.MultiIndex.from_product(
                            [temp_stats.columns.to_list(), temp_stats.index.values]
                        ),
                    ).T

                    # This does the same but works only for data with one column
                    # new_col = temp_stats.columns.any()
                    # temp_stats.insert(loc=0, value=new_col, column=None)
                    # temp_stats.set_index(append=True, keys=None, inplace=True)
                    # temp_stats = temp_stats.swaplevel()
                    # temp_stats = temp_stats.T
                    # temp_stats.reset_index(drop=True, inplace=True)

                statistics[qoi] = temp_stats.copy()
            else:
                # ToDo [Logger] turn print(...) into logger info
                print(
                    f"Skipping computation of summary statistics for quantity "
                    f"of interest '{qoi}'."
                )

        return statistics


class TestUniformMonteCarlo(UQMethod):
    """Only for testing and debugging purposes. Takes only uniform
    distributions."""

    def sample(self, parameter_space: ParameterSpace):
        """Generates random samples that are uniformly distributed within the
        defined intervals"""

        samples = []

        for _ in range(0, self.sample_size):
            sample = dict()
            for p in parameter_space.parameters:
                if isinstance(p.dist, UniformDistribution):
                    sample[p.name] = p.dist.lower_bound + self.RANDOM.random() * (
                        p.dist.upper_bound - p.dist.lower_bound
                    )
                else:
                    raise TypeError(
                        "Distribution of parameter {} should be of type {}".format(
                            p.name, type(UniformDistribution)
                        )
                    )
            samples.append(sample)

        return samples

    def analyze(self, parameter_space: ParameterSpace, model_output, qois, n_samples):
        print("Nothing to analyze.")


# ToDo find better place for this method
def array_to_par_var(sample_parameter_array, parameter_names) -> list:
    """Convert n (number of samples, rows) x m (number of parameters, columns)
    array to list of dictionaries"""

    par_vars = []
    for row in sample_parameter_array:
        par_var = dict()
        for col_idx in range(len(parameter_names)):
            par_var[parameter_names[col_idx]] = row[col_idx]
        par_vars.append(par_var)
    return par_vars
