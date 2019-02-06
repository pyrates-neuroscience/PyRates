# system imports
import os
import sys
import ast
import json
import time
import socket
import argparse
from pathlib import Path

# external imports
import pandas as pd
from pyrates.utility import grid_search


class StreamTee(object):
    """Copy all stdout to a specified file"""
    # Based on https://gist.github.com/327585 by Anand Kunal
    def __init__(self, stream1, stream2fp):
        stream2 = open(stream2fp, "a")
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)


def main(_):
    hostname = socket.gethostname()

    ##################################################
    # Load command line arguments and create logfile #
    ##################################################
    print("")
    print("***LOADING COMMAND LINE ARGUMENTS***")
    # start_arg = time.time()

    global_config = FLAGS.global_config
    local_config = FLAGS.local_config
    local_grid = FLAGS.local_grid
    log_dir = FLAGS.log_dir
    res_dir = FLAGS.res_dir
    grid_name = FLAGS.grid_name

    # Create logfile in Log directory
    logfile = f'{log_dir}/Local_log_{Path(global_config).stem}_{hostname}.log'
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    # Copy all stdout and stderr to logfile
    sys.stdout = StreamTee(sys.stdout, logfile)
    sys.stderr = StreamTee(sys.stderr, logfile)

    # elapsed_arg = time.time() - start_arg
    # print("Done! Elapsed time: {0:.3f} seconds".format(elapsed_arg))

    ###########################
    # Load global config file #
    ###########################
    print("")
    print("***LOADING GLOBAL CONFIG FILE***")
    start_gconf = time.time()

    with open(global_config) as g_conf:
        global_config_dict = json.load(g_conf)

        circuit_template = global_config_dict['circuit_template']
        param_map = global_config_dict['param_map']

        # TODO: Does that work for different, multiple inputs/outputs?
        # Recreate tuple from string representation to use as 'key' in inputs
        inputs = {ast.literal_eval(*global_config_dict['inputs'].keys()):
                  list(*global_config_dict['inputs'].values())}

        # Recreate tuple from list to use as 'values' in outputs
        outputs = {str(*global_config_dict['outputs'].keys()):
                   tuple(*global_config_dict['outputs'].values())}

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

    param_grid = pd.read_csv(local_grid, index_col=0)

    # TODO: Print "Awaiting grid" so that the master can catch this line via stdout and send a new parameter grid

    # TODO: Await a param_grid from stdin to start grid_search()

    # Exclude 'status'-key from param_grid because grid_search() can't handle the additional keyword
    param_grid_arg = param_grid.loc[:, param_grid.columns != "status"]

    elapsed_grid = time.time() - start_grid
    print("Parameter grid loaded. Elapsed time: {0:.3f} seconds".format(elapsed_grid))

    ##########################
    # COMPUTE PARAMETER GRID #
    ##########################
    print("")
    print("***COMPUTING PARAMETER GRID***")
    start_comp = time.time()

    results = grid_search(circuit_template=circuit_template,
                          param_grid=param_grid_arg,
                          param_map=param_map,
                          inputs=inputs,
                          outputs=outputs,
                          sampling_step_size=sampling_step_size,
                          dt=dt,
                          simulation_time=simulation_time)

    elapsed_comp = time.time() - start_comp
    print("Parameter grid computed. Elapsed time: {0:.3f} seconds".format(elapsed_comp))

    #######################
    # CREATE RESULT FILES #
    #######################
    print("")
    print("***CREATING RESULT FILES***")
    start_res = time.time()

    # Columns in results are unsorted
    # Access parameter combinations in param_grid and their corresponding index
    # TODO: Write all results in the same hdf5-file?
    #   Create temp.hfd5. Master assembles all temp files to one big result file?

    # TODO: Implement possibility for customized postprocessing of the result data

    for idx, row in param_grid_arg.iterrows():
        # idx is the index label, e.g. 4,5,6,7 not the absolute index (0,1,2,3)
        res_file = f'{res_dir}/CGS_result_{grid_name}_idx_{idx}.csv'
        params = param_grid_arg.iloc[param_grid_arg.index.get_loc(idx), :]
        # Find corresponding result in results for each parameter combination in param_grid
        result = results.loc[:, (params[0], params[1:])]
        result.index = results.index
        result.to_csv(res_file, index=True)

    elapsed_res = time.time() - start_res
    print("Result files created. Elapsed time: {0:.3f} seconds".format(elapsed_res))

    # TODO: Send output_file back to host if there is no shared memory available


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
        "--local_config",
        type=str,
        default="",
        help="Config file with worker specific instructions. Contains param grid, extra input and commands for further"
             "signal processing"
    )

    parser.add_argument(
        "--local_grid",
        type=str,
        default="",
        help="Path to csv-file with subgrid to compute on the remote machine"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Directory to create local logfile in"
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
