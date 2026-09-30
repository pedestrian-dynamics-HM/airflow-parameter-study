import os
import shutil

import numpy as np
import pandas as pd


def _get_head_path(path, basename):
    """Walks backwards from a given 'path' to the first parent directory called
    'basename'.

    Returns
    ----------
    ret     absolute path to basename

    """

    if basename in path:
        path = os.path.abspath(path)
        if os.path.basename(path) == basename:
            ret = path
        else:
            head, _ = os.path.split(path)
            ret = _get_head_path(head, basename)
    else:
        raise ValueError(f"Cannot find {basename} in {path}.")

    return ret


def remove_output_dirs(parent_directory, dir_prefix):
    """Iterates through all directories stored in parent_directory and removes
    the directories whose names start with dir_prefix.
    """
    parent_directory = os.path.abspath(parent_directory)
    paths = os.listdir(parent_directory)
    paths = [os.path.join(parent_directory, p) for p in paths]
    for p in paths:
        if os.path.isdir(p) and os.path.basename(os.path.normpath(p)).startswith(
            dir_prefix
        ):
            shutil.rmtree(p)

            # ToDo [Logger] turn print(...) into logger info
            print(f"Deleted {p}.")


def get_rand_array(data_array, uncertainty, multiplier, rng, rnd=None):
    """
    Arguments
    ----------
    data_array  np.array
    uncertainty float, gets multiplied by a random number [0, 1]
    multiplier  number of arrays that are created from the data_array, e.g.
                data_array = array([1,2,3]) and multiplier=2 returns
                array(array([1,2,3]), array([1,2,3] + uncertainty * rnd))
    rng         random number generator
    rnd         random number(s)

    """

    if rnd is None:
        rnd = np.round(rng.random(multiplier), 1) * uncertainty
    if data_array.size == 0:
        ret = np.array([[rnd[i]] for i in range(multiplier)])
    else:
        ret = [data_array + rnd[i] for i in range(multiplier)]
    return ret, rnd


def get_sim_output_data_frame(
    n_ids=2, n_runs=2, n_peds=1, n_tsteps=1, redundant_idx=False
):
    """Creates a dataframe that is equivalent to (Vadere/suqc) simulation
    output, i.e. the return value imitates a suqc simulation output (more
    precisely, one value of the dictionary of a suqc simulation output).
    """
    rng = np.random.default_rng(seed=1)

    cols = ["QoI"]
    id_diff = 1000  # difference between output for different samples
    run_diff = 100  # ... runs
    ped_diff = 10  # ... pedestrians
    tstep_diff = 1  # ... time step

    out_0 = 1000  # init value

    index_names = []
    id_idx = np.concatenate(
        [[i] * (n_runs * n_peds * n_tsteps) for i in range(0, n_ids)]
    )
    run_idx = np.concatenate(
        [[i] * (n_peds * n_tsteps) for i in range(0, n_runs)] * n_ids
    )
    ped_idx = np.concatenate(
        [[i] * (n_tsteps) for i in range(0, n_peds)] * (n_ids * n_runs)
    )
    tstep_idx = np.concatenate(
        [[i] for i in range(0, n_tsteps)] * (n_ids * n_runs * n_peds)
    )
    indices = np.array([id_idx, run_idx])
    opt_indices = np.array([])
    data = np.array([])

    # optional output keys (timeStep, pedestrianId)
    if n_tsteps > 1:
        index_names.append("timeStep")
        # some linear increase over time, this could be any other relationship:
        data = np.array(list(range(out_0, n_tsteps * tstep_diff + out_0, tstep_diff)))
        opt_indices = np.array([tstep_idx])

    if n_peds > 1:
        index_names.append("pedestrianId")
        if data.size == 0:
            data = np.array(out_0 + ped_diff * np.round(rng.random(n_peds), 1))
            opt_indices = np.array([ped_idx])
        else:
            data, _ = get_rand_array(data, ped_diff, n_peds, rng=rng)
            data = np.concatenate(data)
            opt_indices = np.array([ped_idx, tstep_idx])

    # redundant indices
    if redundant_idx:
        index_names.append(None)
        red_idx = run_idx.copy()
        indices = np.array([id_idx, run_idx, red_idx])

    if opt_indices.size > 0:
        indices = np.concatenate([indices, opt_indices])

    # mandatory output keys (run_id, id)
    index_names.append("run_id")
    run_data, _ = get_rand_array(data, run_diff, n_runs, rng=rng)
    run_data_mean = np.array(run_data).mean(axis=0)
    run_data = np.concatenate(run_data)

    index_names.append("id")
    id_data, rnd = get_rand_array(run_data, id_diff, n_ids, rng=rng)
    id_data = np.concatenate(id_data)
    id_data_mean, _ = get_rand_array(run_data_mean, id_diff, n_ids, rng=rng, rnd=rnd)
    id_data_mean = np.concatenate(id_data_mean)

    index_names.reverse()
    multi_index = pd.MultiIndex.from_arrays(indices, names=index_names)

    return pd.DataFrame(id_data, columns=cols, index=multi_index), id_data_mean
