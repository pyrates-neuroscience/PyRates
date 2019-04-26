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

    # Load global config file
    #########################
    print("")
    print("***LOADING GLOBAL CONFIG FILE***")
    t0 = time.time()

    with open(config_file) as g_conf:
        global_config_dict = json.load(g_conf)

        circuit_template = global_config_dict['circuit_template']
        param_map = global_config_dict['param_map']
        sampling_step_size = global_config_dict['sampling_step_size']
        dt = global_config_dict['dt']
        simulation_time = global_config_dict['simulation_time']
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

    # LOAD PARAMETER GRID
    #####################
    print("")
    print("***PREPARING PARAMETER GRID***")
    t0 = time.time()

    # Load subgrid into DataFrame
    param_grid = pd.read_hdf(subgrid, key="subgrid")

    # Exclude 'status', 'chunk_idx' and 'err_count' columns from param_grid
    # since grid_search() can't handle the additional keys
    param_grid = param_grid.drop('status', axis=1)
    param_grid = param_grid.drop('chunk_idx', axis=1)
    param_grid = param_grid.drop('err_count', axis=1)
    print(f'Elapsed time: {time.time()-t0:.3f} seconds')

    # COMPUTE PARAMETER GRID #
    ##########################
    print("")
    print("***COMPUTING PARAMETER GRID***")
    t0 = time.time()

    results = grid_search(circuit_template=circuit_template,
                          param_grid=param_grid,
                          param_map=param_map,
                          inputs=inputs,
                          outputs=outputs,
                          sampling_step_size=sampling_step_size,
                          dt=dt,
                          simulation_time=simulation_time)

    out_vars = results.columns.levels[-1]

    print(f'Total parameter grid computation time: {time.time()-t0:.3f} seconds')
    # print(f'Peak memory usage: {m} MB')

    # POST PROCESS DATA AND CREATE RESULT FILES
    ###########################################
    print("")
    print("***POSTPROCESSING AND CREATING RESULT FILES***")
    t0 = time.time()

    # Order results and post processing
    ###################################
    with pd.HDFStore(local_res_file, "w") as store:
        for out_var in out_vars:
            key = out_var.replace(".", "")
            res_lst = []

            # Loop through parameter grid and identify corresponding result column
            for i, idx in enumerate(param_grid.index):
                idx_list = param_grid.iloc[i].values.tolist()
                idx_list.append(out_var)

                result = results.loc[:, tuple(idx_list)].to_frame()
                result.columns.names = results.columns.names
                res_lst.append(result)

                # POSTPROCESSING
                ################

            # Ordered DataFrames with respect to param_grid entries
            result_ordered = pd.concat(res_lst, axis=1)

            # SAVE DATA
            ###########
            store.put(key=key, value=result_ordered)
            # results.to_hdf(local_res_file, key=key)

            # SAVE DATA
            ###########
            # DataFrames
            # with pd.HDFStore(local_res_file, "w") as store:
            #     store.put(key='Data', value=results)
                # store.put(key=f'PSD', value=spec)

            # Other
            # with h5py.File(local_res_file, "a") as file:
                # file.create_dataset("Other/num_peaks", data=peak_lst)

    print(f'Result files created. Elapsed time: {time.time()-t0:.3f} seconds')

    print("")

    print(f'Total elapsed time: {time.time()-t_total:.3f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="/nobackup/spanien1/salomon/ClusterGridSearch/TestData/Test_2/Test_config.json",
        help="Config file with all necessary data to start grid_search() except for parameter grid"
    )

    parser.add_argument(
        "--subgrid",
        type=str,
        default="/nobackup/spanien1/salomon/ClusterGridSearch/TestData/Test_2/Test_subgrid.h5",
        help="Path to csv-file with sub grid to compute on the remote machine"
    )

    parser.add_argument(
        "--local_res_file",
        type=str,
        default="/nobackup/spanien1/salomon/ClusterGridSearch/TestData/Test_2/Test_result.h5",
        help="File to save results to"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)


def postprocessing_1(data_):
    """
    Compute spike frequency based on frequency in PSD with the highest energy

    :param data_:
    :return:
    """
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
    # Return empty DataFrame
    return pd.DataFrame(columns=cols)


def postprocessing_2(data, simulation_time):
    """
    Compute spike frequency based on average number of spikes (local maxima) per second

    :param data:
    :param simulation_time:
    :return:
    """
    import scipy.signal as sp

    cols = data.columns

    np_data = np.array(data.values)

    peaks = sp.argrelextrema(np_data, np.greater)
    num_peaks_temp = int(len(peaks[0]) / simulation_time)

    return pd.DataFrame(num_peaks_temp, index=['num_peaks'], columns=cols)


def postprocessing_3(data, dt):
    """
    Compute spike frequency based on average time between spikes (local maxima)
    :return:
    """
    import scipy.signal as sp

    cols = data.columns

    np_data = np.array(data.values)

    peaks = sp.argrelextrema(np_data, np.greater)
    diff = np.diff(peaks[0])
    diff = np.mean(diff) * dt
    return 1 / diff

