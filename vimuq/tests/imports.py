from argparse import Namespace
import os

from vimuq.tests import _get_head_path

# ---------------------------------------------
# Variables
# ---------------------------------------------
cwd = os.getcwd()
path2project = _get_head_path(cwd, "airflow-uq")

v_id = "test_experiment_vadere"
suffix = ""
v = "vadere"
abs_path2vadere_test = os.path.join(path2project, "vimuq", "tests", "model", "vadere")
v_file = os.path.join(abs_path2vadere_test, "simulator", "test_vadere-console.txt")
scenario_path = os.path.join(
    abs_path2vadere_test, "scenarios", "test_queue_uq_fast.scenario"
)
v_param1 = ["param1", "UniformDistribution", 0, 1]
v_param2 = ["param2", "UniformDistribution", 2.5, 10]
v_qoi = ["test_qoi1", "test_qoi2"]
v_out = os.path.join("..", "output")
run_loc = True
reps = 1
v_avrg_runs = "A"
v_method = "TestUniformMonteCarlo"
v_secondorder = False
v_samples = 2
v_func = "mean"
v_idx_name1 = "pedestrianId"
v_idx_name2 = "timeStep"
v_change_ipl = 999
v_skip_samples = 32

check_endtime = False
check_servicetime = [0.0, 0.0]
check_qoi = "checkqoi"

# ---------------------------------------------
# Experiment definition
# ---------------------------------------------

v_experiment = dict()

v_experiment["argv"] = [
    f"--experimentid {v_id}",
    f"--model {v}",
    f"--modelfile {v_file}",
    f"--scenario {scenario_path}",
    f"--parameter {v_param1[0]} {v_param1[1]} {v_param1[2]} {v_param1[3]}",
    f"--parameter {v_param2[0]} {v_param2[1]} {v_param2[2]} {v_param2[3]}",
    f"--output {v_qoi[0]}",
    f"--output {v_qoi[1]}",
    f"--runlocal {run_loc}",
    f"--repetitions {reps}",
    f"--uqmethod {v_method}",
    f"--secondorder {v_secondorder}",
    f"--samplesize {v_samples}",
    f"--outputpath {v_out}",
    f"--changeinitpathogenload {v_change_ipl}",
    f"--skipsamples {v_skip_samples}",
]

v_experiment["argv_f"] = [
    "--experimentid",
    v_id,
    "--model",
    v,
    "--modelfile",
    v_file,
    "--scenario",
    scenario_path,
    "--parameter",
    v_param1[0],
    v_param1[1],
    str(v_param1[2]),
    str(v_param1[3]),
    "--parameter",
    v_param2[0],
    v_param2[1],
    str(v_param2[2]),
    str(v_param2[3]),
    "--output",
    v_qoi[0],
    "--output",
    v_qoi[1],
    "--runlocal",
    str(run_loc),
    "--repetitions",
    str(reps),
    "--uqmethod",
    v_method,
    "--secondorder",
    str(v_secondorder),
    "--samplesize",
    str(v_samples),
    "--outputpath",
    v_out,
    "--changeinitpathogenload",
    str(v_change_ipl),
    "--skipsamples",
    str(v_skip_samples),
]

v_experiment["namespace"] = Namespace(
    experimentid=v_id,
    model=v,
    modelfile=v_file,
    output=v_qoi,
    outputpath=v_out,
    parameter=[
        [v_param1[0], v_param1[1], str(v_param1[2]), str(v_param1[3])],
        [v_param2[0], v_param2[1], str(v_param2[2]), str(v_param2[3])],
    ],
    repetitions=reps,
    runlocal=str(run_loc),
    samplesize=v_samples,
    scenario=scenario_path,
    uqmethod=v_method,
    secondorder=str(v_secondorder),
    changeinitpathogenload=v_change_ipl,
    skipsamples=v_skip_samples,
)

# namespace has no attribute 'copy'
v_experiment["args"] = Namespace(
    experimentid=v_id,
    model=v,
    modelfile=v_file,
    output=v_qoi,
    outputpath=v_out,
    parameter=[
        [v_param1[0], v_param1[1], v_param1[2], v_param1[3]],
        [v_param2[0], v_param2[1], v_param2[2], v_param2[3]],
    ],
    repetitions=reps,
    runlocal=run_loc,
    samplesize=v_samples,
    scenario=scenario_path,
    uqmethod=v_method,
    secondorder=v_secondorder,
    changeinitpathogenload=v_change_ipl,
    skipsamples=v_skip_samples,
)

# another experiment definition for a python based model, here use the ishigami
# test function
pm_id = "test_experiment_pymodel"
pm = "pymodel"
pm_experiment = dict()
pm_file = os.path.join(
    path2project, "vimuq", "tests", "model", "pymodel", "ishigami.py"
)
pm_param1 = ["x1", "UniformDistribution", -3.14159265359, 3.14159265359]
pm_param2 = ["x2", "UniformDistribution", -3.14159265359, 3.14159265359]
pm_param3 = ["x3", "UniformDistribution", -3.14159265359, 3.14159265359]
pm_qoi = ["Y"]
pm_method = "SALibSaltelliSobolSequence"
pm_samples = 1024
pm_out = os.path.join("..", "output")

pm_experiment["argv"] = [
    f"--experimentid {pm_id}",
    f"--model {pm}",
    f"--modelfile {pm_file}",
    f"--parameter {pm_param1[0]} {pm_param1[1]} {pm_param1[2]} {pm_param1[3]}",
    f"--parameter {pm_param2[0]} {pm_param2[1]} {pm_param2[2]} {pm_param2[3]}",
    f"--parameter {pm_param3[0]} {pm_param3[1]} {pm_param3[2]} {pm_param3[3]}",
    f"--output {pm_qoi[0]}",
    f"--uqmethod {pm_method}",
    f"--samplesize {pm_samples}",
    f"--outputpath {pm_out}",
]

# ---------------------------------------------
# Experiment analysis
# ---------------------------------------------

v_analysis = dict()

v_analysis["argv"] = [
    f"--experimentid {v_id}",
    f"--experimentsuffix {suffix}",
    f"--outputpath {v_out}",
    f"--quantitiesofinterest {v_qoi[0]} {v_qoi[1]}",
    f"--averagerepetitions {v_avrg_runs}",
    f"--applyfunction {v_qoi[0]} {v_func} {v_idx_name1} {v_idx_name2}",
    f"--checkendtimes {check_endtime}",
    f"--checkservicetimes {check_servicetime}",
    f"--controlqoi {check_qoi}",
    f"--samples {v_samples}",
]

# argv formatted
v_analysis["argv_f"] = [
    "--experimentid",
    v_id,
    "--experimentsuffix",
    suffix,
    "--outputpath",
    v_out,
    "--quantitiesofinterest",
    v_qoi[0],
    v_qoi[1],
    "--averagerepetitions",
    v_avrg_runs,
    "--applyfunction",
    v_qoi[0],
    v_func,
    v_idx_name1,
    v_idx_name2,
    "--checkendtimes",
    check_endtime,
    "--checkservicetimes",
    str(check_servicetime[0]),
    str(check_servicetime[1]),
    "--controlqoi",
    check_qoi,
    "--samples",
    str(v_samples),
]

v_analysis["namespace"] = Namespace(
    experimentid=v_id,
    experimentsuffix=suffix,
    outputpath=v_out,
    quantitiesofinterest=[v_qoi[0], v_qoi[1]],
    averagerepetitions=v_avrg_runs,
    applyfunction=[[v_qoi[0], v_func, v_idx_name1, v_idx_name2]],
    checkendtimes=str(check_endtime),
    checkservicetimes=[check_servicetime[0], check_servicetime[1]],
    controlqoi=check_qoi,
    samples=v_samples,
)
