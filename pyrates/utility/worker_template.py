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
import ast
import json
import time
import argparse

# external imports
import numpy as np
import pandas as pd
from pyrates.utility.grid_search import grid_search


def main(_):
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

    print(f'Elapsed time: {time.time()-t0:.3f} seconds')

    # Load grid search configuration parameters from config file
    ############################################################
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
        # sampling_step_size
        try:
            sampling_step_size = global_config_dict['sampling_step_size']
        except KeyError:
            sampling_step_size = dt

        try:
            backend = global_config_dict['backend']
        except KeyError:
            backend = 'numpy'

        try:
            inputs_temp = global_config_dict['inputs']
            if inputs_temp:
                inputs = {}
                for key, value in inputs_temp.items():
                    inputs[ast.literal_eval(key)] = list(value)
            else:
                inputs = {}
        except KeyError:
            inputs = {}

        try:
            outputs_temp = global_config_dict['outputs']
            if outputs_temp:
                outputs = {}
                for key, value in outputs_temp.items():
                    outputs[str(key)] = tuple(value)
            else:
                outputs = {}
        except KeyError:
            outputs = {}

    print(f'Elapsed time: {time.time()-t0:.3f} seconds')

    # Load parameter subgrid from subgrid file
    ##########################################
    print("")
    print("***PREPARING PARAMETER GRID***")
    t0 = time.time()

    param_grid = pd.read_hdf(subgrid, key="subgrid")

    # grid_search() can't handle additional columns in the parameter grid
    try:
        param_grid = param_grid.drop(['status', 'chunk_idx', 'err_count'], axis=1)
    except KeyError:
        pass

    print(f'Elapsed time: {time.time()-t0:.3f} seconds')

    # Compute parameter subgrid using grid_search
    #############################################
    print("")
    print("***COMPUTING PARAMETER GRID***")
    t0 = time.time()

    # grid_search returns an unsorted DataFrame yielding results for all parameter combinations in param_grid
    results, _t, _ = grid_search(
        circuit_template=circuit_template,
        param_grid=param_grid,
        param_map=param_map,
        simulation_time=simulation_time,
        dt=dt,
        sampling_step_size=sampling_step_size,
        permute_grid=False,
        inputs=inputs,
        outputs=outputs,
        init_kwargs={
            'backend': backend,
            'vectorization': 'nodes'
        },
        profile='t')

    out_vars = results.columns.levels[-1]

    print(f'Total parameter grid computation time: {time.time()-t0:.3f} seconds')

    # Post process results and write data to local result file
    ##########################################################
    print("")
    print("***POSTPROCESSING AND CREATING RESULT FILES***")
    t0 = time.time()

    with pd.HDFStore(local_res_file, "w") as store:
        for out_var in out_vars:
            key = out_var.replace(".", "")
            res_lst = []

            # Order results according to rows in parameter grid
            ###################################################
            # Iterate over rows in param_grid and use its values to index columns in results
            for i, column_values in enumerate(param_grid.values):
                result = results.loc[:, tuple([column_values, out_var])]
                result.columns.names = results.columns.names
                res_lst.append(result)

            # Concatenate all DataFrame in res_lst to one global ordered DataFrame
            result_ordered = pd.concat(res_lst, axis=1)

            # Postprocess ordered results (optional)
            ########################################

            # Write DataFrames to local result file
            ######################################
            store.put(key=key, value=result_ordered)

    # TODO: Copy local result file back to master if needed

    print(f'Result files created. Elapsed time: {time.time()-t0:.3f} seconds')
    print("")
    print(f'Total elapsed time: {time.time()-t_total:.3f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="/nobackup/spanien1/salomon/CGS/Holgado/GeneticClusterFit_diseased_1/Config/DefaultConfig_3.json",
        help="Config file with all necessary data to start grid_search() except for parameter grid"
    )

    parser.add_argument(
        "--subgrid",
        type=str,
        default="/nobackup/spanien1/salomon/CGS/Holgado/GeneticClusterFit_diseased_1/Grids/Subgrids/DefaultGrid_3/tiber/tiber_Subgrid_0.h5",
        help="Path to csv-file with sub grid to compute on the remote machine"
    )

    parser.add_argument(
        "--local_res_file",
        type=str,
        default="//data/hu_salomon/Documents/CGSWorkerTests/test/test_result.h5",
        help="hdf5-file to save results to"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)


def postprocessing_1(data_):
    """Compute spike frequency based on frequency in PSD with the highest energy"""

    from pyrates.utility import plot_psd
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

