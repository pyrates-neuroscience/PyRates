# # system imports
import argparse
import sys
import ast
import json
import socket
import os
from pathlib import Path

# external imports
import pandas as pd
# TODO: Loading pyrates takes very long, also on the remote hosts. Makes the computation extremely long, especially
#   when remote script is called multiple times, everytime loading pyrates again
from pyrates.utility import grid_search


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(_):
    # TODO: Create more outputs to track the progress in the logfile
    # TODO: Add timestamps

    hostname = socket.gethostname()

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
    sys.stdout = Logger(logfile)
    sys.stderr = Logger(logfile)

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

    param_grid = pd.read_csv(local_grid, index_col=0)
    param_idx = param_grid.index.tolist()

    # TODO: Print "Awaiting grid" so that the master can catch this line via stdout and send a new parameter grid

    # TODO: Await a param_grid from stdin to start grid_search()

    # Exclude 'status'-key from param_grid because grid_search() can't handle the additional keyword
    param_grid_arg = param_grid.loc[:, param_grid.columns != "status"]

    results = grid_search(circuit_template=circuit_template,
                          param_grid=param_grid_arg,
                          param_map=param_map,
                          inputs=inputs,
                          outputs=outputs,
                          sampling_step_size=sampling_step_size,
                          dt=dt,
                          simulation_time=simulation_time)

    for column in range(len(results.columns)):
        # TODO: Write name of the column multiindex to file, not only the values
        results.iloc[:, column].to_csv(f'{res_dir}/CGS_result_{grid_name}_idx_{param_idx[column]}.csv', index=True)

    # TODO: Write name of used config file and parameter combination to each result file

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
