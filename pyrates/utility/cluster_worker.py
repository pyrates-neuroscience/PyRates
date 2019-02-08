# system imports
import sys
import ast
import json
import time
import argparse

# external imports
import pandas as pd
from pyrates.utility import grid_search

# TODO: Add additional prints for more detailed logfile


def main(_):
    ##################################################
    # Load command line arguments and create logfile #
    ##################################################
    print("")
    print("***LOADING COMMAND LINE ARGUMENTS***")
    start_arg = time.time()

    global_config = FLAGS.global_config
    subgrid = FLAGS.subgrid
    res_dir = FLAGS.res_dir
    grid_name = FLAGS.grid_name

    elapsed_arg = time.time() - start_arg
    print("Done! Elapsed time: {0:.3f} seconds".format(elapsed_arg))

    ###########################
    # Load global config file #
    ###########################
    print("")
    print("***LOADING GLOBAL CONFIG FILE***")
    start_gconf = time.time()

    with open(global_config) as g_conf:
        global_config_dict = json.load(g_conf)

        # Recreate tuple from string/list to use as key/values in inputs/outputs since pure tuples cannot be saved in
        # json formatted files
        inputs = {ast.literal_eval(*global_config_dict['inputs'].keys()):
                  list(*global_config_dict['inputs'].values())}
        outputs = {str(*global_config_dict['outputs'].keys()):
                   tuple(*global_config_dict['outputs'].values())}
        circuit_template = global_config_dict['circuit_template']
        param_map = global_config_dict['param_map']
        sampling_step_size = global_config_dict['sampling_step_size']
        dt = global_config_dict['dt']
        simulation_time = global_config_dict['simulation_time']

    elapsed_gconf = time.time() - start_gconf
    print("Global config loaded. Elapsed time: {0:.3f} seconds".format(elapsed_gconf))

    #########################
    # LOAD PARAMETER GRID #
    #########################
    print("")
    print("***LOADING PARAMETER GRID***")
    start_grid = time.time()

    # Load subgrid into DataFrame
    param_grid = pd.read_csv(subgrid, index_col=0)

    # Exclude 'status'-key from param_grid since grid_search() can't handle the additional keyword
    param_grid = param_grid.loc[:, param_grid.columns != "status"]

    elapsed_grid = time.time() - start_grid
    print("Parameter grid loaded. Elapsed time: {0:.3f} seconds".format(elapsed_grid))

    ##########################
    # COMPUTE PARAMETER GRID #
    ##########################
    print("")
    print("***COMPUTING PARAMETER GRID***")
    start_comp = time.time()

    results = grid_search(circuit_template=circuit_template,
                          param_grid=param_grid,
                          param_map=param_map,
                          inputs=inputs,
                          outputs=outputs,
                          sampling_step_size=sampling_step_size,
                          dt=dt,
                          simulation_time=simulation_time)

    elapsed_comp = time.time() - start_comp
    print("Parameter grid computed. Elapsed time: {0:.3f} seconds".format(elapsed_comp))

    ############################################
    # POSTPROCESS DATA AND CREATE RESULT FILES #
    ############################################
    print("")
    print("***POSTPROCESSING AND CREATING RESULT FILES***")
    start_res = time.time()

    for idx, row in param_grid.iterrows():
        # idx is the absolute index label, e.g. 4,5,6,7 not the relative index of the current DataFrame (0,1,2,3)
        res_file = f'{res_dir}/CGS_result_{grid_name}_idx_{idx}.csv'

        # Get parameter combination of the current index and the corresponding result data
        params = param_grid.iloc[param_grid.index.get_loc(idx), :]
        result = results.loc[:, (params[0], params[1:])]

        ##################
        # POSTPROCESSING #
        ##################
        result = postprocessing(result)

        result.index = results.index
        result.to_csv(res_file, index=True)

    elapsed_res = time.time() - start_res
    print("Result files created. Elapsed time: {0:.3f} seconds".format(elapsed_res))


def postprocessing(data):
    # TODO: Implement possibility for customized postprocessing of the result data
    # Read data for each column in results, process data and write it back to the column
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--global_config",
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
        "--res_dir",
        type=str,
        default="",
        help="Directory to save result files to"
    )

    parser.add_argument(
        "--grid_name",
        type=str,
        default="",
        help="Name of the parameter grid currently being computed"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)
