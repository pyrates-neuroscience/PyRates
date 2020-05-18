# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network models and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

# system imports
import os
import sys
import json
import time
import argparse
import warnings

# external imports
from numba import njit, config
import numpy as np
import pandas as pd
import scipy.signal as sp
from pyrates.utility.grid_search import grid_search


def cgs_postprocessing(data: np.array):
    """Post processing function that is applied to every model parametrization output.

    Parameters
    ----------
    data
        Simulated model output for a single model parametrization

    Returns
    -------
    np.array
        Post processed model output

    """

    # Add customized post processing here
    # e.g. Computation of power spectral density of the time signal:
    #
    # from scipy.signal import welch
    # dt = data[1] - data[0]
    # f, p  = welch(data.values, fs=1 / dt, axis=0, **kwargs)
    # processed_data = p

    # Placeholder
    processed_data = data

    return processed_data


# Don't make any changes below #
################################

def main(_):
    # tf.config.set_soft_device_placement(True)

    config.THREADING_LAYER = 'omp'

    # Disable general warnings
    warnings.filterwarnings("ignore")

    # disable TF-gpu warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    t_total = time.time()

    # Load command line arguments and create logfile
    ################################################
    print("")
    print("***LOADING COMMAND LINE ARGUMENTS***")
    t0 = time.time()

    config_file = FLAGS.config_file
    subgrid = FLAGS.subgrid
    local_res_file = FLAGS.local_res_file
    build_dir = FLAGS.build_dir

    print(f'Elapsed time: {time.time() - t0:.3f} seconds')

    # Load global config file
    #########################
    print("")
    print("***LOADING GLOBAL CONFIG FILE***")
    t0 = time.time()

    with open(config_file) as g_conf:
        global_config_dict = json.load(g_conf)
        circuit_template = global_config_dict['circuit_template']
        param_map = global_config_dict['param_map']
        dt = global_config_dict['dt']
        simulation_time = global_config_dict['simulation_time']

        # Optional parameters
        #####################
        try:
            sampling_step_size = global_config_dict['sampling_step_size']
        except KeyError:
            sampling_step_size = dt

        try:
            inputs = global_config_dict['inputs']
        except KeyError:
            inputs = {}

        try:
            outputs = global_config_dict['outputs']
        except KeyError:
            outputs = {}

        try:
            init_kwargs = global_config_dict['init_kwargs']
        except KeyError:
            init_kwargs = {}

        print(f'Elapsed time: {time.time() - t0:.3f} seconds')

        # LOAD PARAMETER GRID
        #####################
        print("")
        print("***PREPARING PARAMETER GRID***")
        t0 = time.time()

        # Load subgrid into DataFrame
        param_grid = pd.read_hdf(subgrid, key="subgrid")

        # Drop all columns that don't contain a parameter map value (e.g. status, chunk_idx, err_count) since
        # grid_search() can't handle additional columns
        param_grid = param_grid[list(param_map.keys())]
        print(f'Elapsed time: {time.time() - t0:.3f} seconds')

        # COMPUTE PARAMETER GRID
        ########################
        print("")
        print("***COMPUTING PARAMETER GRID***")
        t0 = time.time()

        results, result_map, t_ = grid_search(
            circuit_template=circuit_template,
            param_grid=param_grid,
            param_map=param_map,
            simulation_time=simulation_time,
            dt=dt,
            sampling_step_size=sampling_step_size,
            permute_grid=False,
            inputs=inputs,
            outputs=outputs.copy(),
            init_kwargs=init_kwargs,
            profile='t',
            build_dir=build_dir,
            njit=True,
            parallel=False)

    print(f'Total parameter grid computation time: {time.time()-t0:.3f} seconds')

    # Post process results and write data to local result file
    ##########################################################
    print("")
    print("***POSTPROCESSING AND CREATING RESULT FILES***")
    t0 = time.time()

    processed_results = pd.DataFrame(data=None, columns=results.columns, index=results.index)
    for idx, circuit in enumerate(result_map.iterrows()):
        circ_idx = result_map.loc[(result_map == tuple(circuit[1].values)).all(1), :].index
        processed_results[circ_idx] = cgs_postprocessing(results[circ_idx].to_numpy())

    with pd.HDFStore(local_res_file, "w") as store:
        store.put(key='results', value=results)
        store.put(key='result_map', value=result_map)

    # TODO: Copy local result file back to master if needed

    print(f'Result files created. Elapsed time: {time.time()-t0:.3f} seconds')
    print("")
    print(f'Total elapsed time: {time.time()-t_total:.3f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="/nobackup/spanien1/salomon/CGS/Benchmark_jup/Config/DefaultConfig_5.json",
        help="File to load grid_search configuration parameters from"
    )

    parser.add_argument(
        "--subgrid",
        type=str,
        default="/nobackup/spanien1/salomon/CGS/Benchmark_jup/Grids/Subgrids/DefaultGrid_0/animals/cgs_test_grid.h5",
        help="File to load parameter grid from"
    )

    parser.add_argument(
        "--local_res_file",
        type=str,
        default="/nobackup/spanien1/salomon/WorkerTestData/holgado_subgrid/test_result.h5",
        help="File to save results to"
    )

    parser.add_argument(
        "--build_dir",
        type=str,
        default=os.getcwd(),
        help="Custom PyRates build directory"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)


def postprocessing_1(data_):
    """Compute spike frequency based on frequency in PSD with the highest energy"""

    from pyrates.utility.visualization import plot_psd
    import matplotlib.pyplot as plt

    # Store columns for later reconstruction
    cols = data_.columns

    # Plot_psd expects only the out_var as column value
    index_dummy = pd.Index([cols.values[-1][-1]])
    data_.columns = index_dummy
    if not any(np.isnan(data_.values)):
        _ = plot_psd(data_, tmin=1.0, show=False)
        pow_ = plt.gca().get_lines()[-1].get_ydata()
        freqs = plt.gca().get_lines()[-1].get_xdata()
        plt.close()
        max_freq = freqs[np.argmax(pow_)]
        freq_pow = np.max(pow_)
        temp = [max_freq, freq_pow]
        psd = pd.DataFrame(temp, index=['max_freq', 'freq_pow'], columns=cols)
        data_.columns = cols
        return psd

    data_.columns = cols

    # Return empty DataFrame if data contains NaN
    return pd.DataFrame(columns=cols)


def postprocessing_2(data, simulation_time):
    """Compute spike frequency based on average number of spikes (local maxima) per second"""

    import scipy.signal as sp

    cols = data.columns

    np_data = np.array(data.values)

    peaks = sp.argrelextrema(np_data, np.greater)
    num_peaks_temp = int(len(peaks[0]) / simulation_time)

    return pd.DataFrame(num_peaks_temp, index=['num_peaks'], columns=cols)


def postprocessing_3(data, dt):
    """Compute spike frequency based on average time between spikes (local maxima)"""

    import scipy.signal as sp

    cols = data.columns

    np_data = np.array(data.values)

    peaks = sp.argrelextrema(np_data, np.greater)
    diff = np.diff(peaks[0])
    diff = np.mean(diff) * dt
    return 1 / diff

