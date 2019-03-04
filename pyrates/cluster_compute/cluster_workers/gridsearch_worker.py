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
from pyrates.utility.grid_search import grid_search_2


def main(_):
    # disable TF-gpu warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    t_total = time.time()

    ##################################################
    # Load command line arguments and create logfile #
    ##################################################
    print("")
    print("***LOADING COMMAND LINE ARGUMENTS***")
    t0 = time.time()

    config_file = FLAGS.config_file
    subgrid = FLAGS.subgrid
    res_file = FLAGS.res_file

    print(f'Elapsed time: {time.time()-t0:.3f} seconds')

    ###########################
    # Load global config file #
    ###########################
    print("")
    print("***LOADING GLOBAL CONFIG FILE***")
    t0 = time.time()

    with open(config_file) as g_conf:
        global_config_dict = json.load(g_conf)
        inputs = {ast.literal_eval(*global_config_dict['inputs'].keys()):
                  list(*global_config_dict['inputs'].values())}
        outputs = {str(*global_config_dict['outputs'].keys()):
                   tuple(*global_config_dict['outputs'].values())}
        circuit_template = global_config_dict['circuit_template']
        param_map = global_config_dict['param_map']
        sampling_step_size = global_config_dict['sampling_step_size']
        dt = global_config_dict['dt']
        simulation_time = global_config_dict['simulation_time']

    print(f'Elapsed time: {time.time()-t0:.3f} seconds')

    #########################
    # LOAD PARAMETER GRID #
    #########################
    print("")
    print("***LOADING PARAMETER GRID***")
    t0 = time.time()

    # Load subgrid into DataFrame
    param_grid = pd.read_hdf(subgrid, key='Data')

    # Exclude 'status'- and 'worker'-keys from param_grid since grid_search() can't handle the additional keywords
    param_grid = param_grid.loc[:, param_grid.columns != "status"]
    param_grid = param_grid.loc[:, param_grid.columns != "worker"]

    print(f'Total parameter grid computation time: {time.time()-t0:.3f} seconds')

    ##########################
    # COMPUTE PARAMETER GRID #
    ##########################
    print("")
    print("***COMPUTING PARAMETER GRID***")
    t0 = time.time()

    results, t, = grid_search_2(circuit_template=circuit_template,
                                param_grid=param_grid,
                                param_map=param_map,
                                inputs=inputs,
                                outputs=outputs,
                                sampling_step_size=sampling_step_size,
                                dt=dt,
                                simulation_time=simulation_time,
                                timestamps=True,
                                prec=3)

    print(f'Total parameter grid computation time: {time.time()-t0:.3f} seconds')
    # print(f'Peak memory usage: {m} MB')

    ############################################
    # POSTPROCESS DATA AND CREATE RESULT FILES #
    ############################################
    print("")
    print("***POSTPROCESSING AND CREATING RESULT FILES***")
    t0 = time.time()

    with pd.HDFStore(res_file, "a") as store:
        for col in range(len(results.columns)):
            result = results.iloc[:, col]
            idx_label = result.name[:-1]
            idx = param_grid[(param_grid.values == idx_label).all(1)].index
            result = result.to_frame()
            result.columns.names = results.columns.names

            # Create single result file for each result
            # res_file = f'{res_dir}/CGS_result_{grid_name}_idx_{idx[0]}.h5'

            # POSTPROCESSING
            ################
            # spec = postprocessing(result)

            # SAVE DATA
            ###########
            store.put(key=f'GridIndex/Idx_{idx[0]}/Data', value=result)
            # store.put(key=f'GridIndex/Idx_{idx[0]}/Spec', value=spec)

    print("")
    print(f'Result files created. Elapsed time: {time.time()-t0:.3f} seconds')
    print("")
    print(f'Total elapsed time: {time.time()-t_total:.3f} seconds')


def postprocessing(data):
    # Example: Compute PSD from results
    from pyrates.utility import plot_psd
    import matplotlib.pyplot as plt

    # Store columns for later reconstruction
    cols = data.columns

    # Plot_psd expects only the out_var as column value
    index_dummy = pd.Index([cols.values[-1][-1]])
    data.columns = index_dummy
    if not any(np.isnan(data.values)):
        _ = plot_psd(data, tmin=1.0, show=False)
        pow_ = plt.gca().get_lines()[-1].get_ydata()
        freqs = plt.gca().get_lines()[-1].get_xdata()
        plt.close()
        psd = pd.DataFrame(pow_, index=freqs, columns=cols)
        data.columns = cols
        return psd

    # Reconstruct columns
    data.columns = cols
    return pd.DataFrame(columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="Config file with all necessary data to start grid_search() except for parameter grid"
    )

    parser.add_argument(
        "--subgrid",
        type=str,
        default="",
        help="Path to csv-file with subgrid to compute on the remote machine"
    )

    parser.add_argument(
        "--res_file",
        type=str,
        default="",
        help="File to save results to"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)
