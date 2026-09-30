"""Command-line utility for vimuq analysis"""
import argparse
import os
import sys
import vimuq
from vimuq.uq.uq_experiment import UQExperiment, Stage
from vimuq.util.argv import split_argument_list, _str_to_bool
from vimuq.util.post_process_application import PostProcessApplication


def _get_parser():
    parser = argparse.ArgumentParser(
        description="""Loads an existing experiment and applies analysis
        methods""",
    )
    parser.add_argument(
        "--experimentid",
        type=str,
        default="unknown",
        help="""An identifier for the experiment. Each id
                        should only exist once.""",
    )
    parser.add_argument(
        "--experimentsuffix",
        type=str,
        default=None,
        help="""Identifier for experiments that have been executed multiple 
        times (and therefore do not have a unique experiment id).""",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default=os.path.join("..", "output"),
        help="""Relative path to the output parent folder. The experiment
        summary and simulation results will be stored as sub directories.""",
    )
    parser.add_argument(
        "--averagerepetitions",
        type=str,
        default="A",
        choices=["A", "B", "C", "D"],
        help="""Indicates whether repeated vadere simulations (suqc runs) with
        different seeds are averaged.
        Depending on the applied functions, it can make a difference when the
        data is averaged. For example, applying a function like 'min' and
        then averaging over all runs (repetitions) yields a different result 
        than the other way round since these operations are not commutative.
        Exceptions: The functions 'sum' and 'mean' could be applied either
        before or after averaging over the simulation output.
        
        If 'A', repetitions are averaged before any functions are applied.
        
        If 'B', repetitions are averaged after functions are applied and before
        UQ methods are applied.
        
        If 'C', repetitions are averaged after UQ methods are applied.
        
        If 'D', the simulation output all repetitions are considered
        individually.""",
    )
    parser.add_argument(
        "--applyfunction",
        type=str,
        nargs="*",
        action="append",
        default=list(),
        # metavar=("qoi", "function", "group_by_indices"),
        help="""Quantity of interest that will be post-processed with any 
        function that can be applied to pandas DataFrames,
        {valid_functions},
        before the analysis is performed. Postprocessing creates a new QoI as 
        DataFrame, which keeps the indices that are defined as group_by_indices.
        Arguments are QoI, function, group_by_indices, e.g.:
        '--applyfunction pedIdKey_degExposure max id' or 
        '--applyfunction timeStepPedIdKey_degExposure mean id pedestrianId' or
        
        The functions can be extended by an operator (<, >, >=, <=) and a 
        numeric value. The function, operator, and the value are connected by an
        underscore. E.g., mean_>=_100, that is the evaluation ignores all 
        values >= 100:       
        '--applyfunction pedIdKey_degExposure mean_>=_100 id'
        
        Deprecated:
        'z_' indicates that zero values are ignored.
        
        Use the QoI without any file extension.
        """.format(
            valid_functions=PostProcessApplication.VALID_FUNCS
        ),
    )
    parser.add_argument(
        "--quantitiesofinterest",
        type=str,
        nargs="*",
        # action="append",
        default=list(),
        help="""Quantities of interest that will be analyzed (for which one 
        obtains sensitivity indices). If not given / an empty list is provided,
        all simulation outputs will be considered. Use the QoI without any file 
        extension.""",
    )
    parser.add_argument(
        "--controlqoi",
        type=str,
        default="pedIdKey_pedEndTime",
        help="""Quantity of interest for pre analysis checks. Use the QoI
        without any file extension.""",
    )
    parser.add_argument(
        "--checkendtimes",
        type=str,
        default="False",
        choices=["True", "False"],
        help="""Indicates whether Vadere simulation results are checked before
        analysis. Simulations are considered invalid/failed, if any pedestrian
        does not leave the topography before simulation ends. This requires a
        control quantity of interest.""",
    )
    parser.add_argument(
        "--checkservicetimes",
        nargs=2,
        type=float,
        metavar=("mean_service_time", "deviation_service_time"),
        help="""Checks simulations for validity. A simulation is considered 
        invalid/failed if pedestrians get stuck, that is the service times do
        not comply with the predefined waiting times. The time between two 
        subsequent exits (pedestrian Id does not matter here) is outside of 
        bounds mean_service_time +- deviation_service_time * mean_service_time.
        This applies only to scenarios with service units. This requires a 
        control quantity of interest.""",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="""Number of samples, N = 2 ** n, to be processed. The Sobol 
        sequence allows to use the first N * (2 * D + 2) samples if second order 
        is calculated or N * (D + 2) samples if only first order is calculated,
        where D is the number of parameters. N must be a power of two!
        By default, use all available samples.""",
    )

    return parser


def _get_arguments(argv):
    parser = _get_parser()
    args = parser.parse_args(argv)

    # convert str input to boolean
    args.checkendtimes = _str_to_bool(args.checkendtimes)

    return args


def main(argv):
    """Main process for analyzing a numerical experiment in stage POST_RUN.

    Arguments
    ----------
    argv :  list of command line arguments. Accepts the following formats:
            a) Typically, if main is called by a script:
            argv = ['--opt1 arg', '--opt2 arg1 arg2', ...]
            b) Typically, if main is called from the command line:
            argv = ['--opt1' 'arg', '--opt2', 'arg1', 'arg2', ...]

    """

    argv = split_argument_list(argv)

    args = _get_arguments(argv)

    path2dumped_experiment = vimuq.get_experiment_dir_name(
        dir_name=os.path.join(args.outputpath, args.experimentid),
        suffix=args.experimentsuffix,
    )
    experiment = UQExperiment.read_experiment(
        path2dumped_experiment, stage=Stage.POST_RUN
    )
    experiment.path2output = path2dumped_experiment

    app_functions = []
    if args.applyfunction:
        [
            app_functions.append(
                PostProcessApplication(qoi=arg[0], function=arg[1], indices=arg[2:])
            )
            for arg in args.applyfunction
        ]

    experiment.pre_analysis_clean_up()

    experiment.check_simulation_output(
        control_qoi=args.controlqoi,
        end_times=args.checkendtimes,
        service_times=args.checkservicetimes,
    )

    experiment.analyze(
        qois=args.quantitiesofinterest,
        qoi_app_functions=app_functions,
        average_repetitions=args.averagerepetitions,
        n_samples=args.samples,
    )

    experiment.save()


if __name__ == "__main__":
    main(sys.argv[1:])
