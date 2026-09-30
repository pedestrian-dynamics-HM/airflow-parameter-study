__version__ = "0.1"


def _get_experiment_pkl_filename(
    stage, prefix="uq_experiment", suffix="", avrg_technique="", n_samples=None
):
    stage_name = stage.name.lower()
    if suffix != "":
        suffix = "-" + suffix
    if avrg_technique != "":
        avrg_technique = "_" + avrg_technique
    if n_samples:
        n_samples = "_" + str(n_samples)
    else:
        n_samples = ""

    return f"{prefix}_{stage_name}{avrg_technique}{n_samples}{suffix}.pkl"


def get_experiment_dir_name(dir_name, suffix):
    if (suffix is None) or (suffix == ""):
        return dir_name
    else:
        return dir_name + "-" + suffix
