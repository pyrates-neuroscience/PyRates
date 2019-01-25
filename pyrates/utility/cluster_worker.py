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


def dummy():
    x = 0
    for i in range(200000000):
        x = x + 1


def main(_):
    compute_id = FLAGS.compute_id
    hostname = socket.gethostname()

    # Create logfile in Log directory
    logfile = f'{FLAGS.log_path}/Local_log_{Path(FLAGS.global_config).stem}_{hostname}.log'

    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    # Copy all stdout and stderr to logfile
    sys.stdout = Logger(logfile)
    sys.stderr = Logger(logfile)

    with open(FLAGS.global_config) as file:
        param_dict = json.load(file)
        try:
            circuit_template = param_dict['circuit_template']
            param_map = param_dict['param_map']

            # TODO: Does that work for different, multiple inputs/outputs?
            # Recreate tuple from string representation to use as 'key' in inputs
            inputs = {ast.literal_eval(*param_dict['inputs'].keys()):
                      list(*param_dict['inputs'].values())}

            # Recreate tuple from list to use as 'values' in outputs
            outputs = {str(*param_dict['outputs'].keys()):
                       tuple(*param_dict['outputs'].values())}

            sampling_step_size = param_dict['sampling_step_size']
            dt = param_dict['dt']
            simulation_time = param_dict['simulation_time']
            result_path = FLAGS.result_path
        except KeyError as err:
            # If config_file does not contain any of the necessary keys
            print("KeyError:", err)
            return

    # Recreate param_grid{} from its string representation and create a DataFrame from it
    param_grid = pd.DataFrame(ast.literal_eval(FLAGS.param_grid_arg))

    # Recreate fetched indices
    param_grid_row_idx = param_grid.index.tolist()

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

    # TODO: Write name of used config file and parameter combination to each result file
    # Write each result to a separate file
    # for col_idx, series in results.iteritems():

    # for idx in range(len(results.columns)):
    #     grid_idx = param_grid_row_idx[idx]
    #     temp = pd.DataFrame(results.iloc[:, idx])
    #     print(temp)
        # file = f'/data/hu_salomon/Documents/ClusterGridSearch/Results/test_gridIdx_{grid_idx}.csv'
        # print(f'Writing results to: {file}')
        # temp.to_csv(file, index=False)
    # TODO: Create output_file for each result in results

    # TODO: Send output_file back to host if there is no shared memory space available


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--global_config",
        type=str,
        default="",
        help="Config file with all necessary data to start grid_search() except for param_grid"
    )

    parser.add_argument(
        "--local_config",
        type=str,
        default="",
        help="Config file with worker specific instructions. Contains param grid, extra input and commands for further"
             "signal processing"
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="",
        help="Directory to create local logfile in"
    )

    parser.add_argument(
        "--compute_id",
        type=str,
        default="",
        help="Unique ID of the whole parameter computation to differentiate created files from different computations"
    )

    parser.add_argument(
        "--res_dir",
        type=str,
        default="",
        help="Shared directory or directory on the master to save/copy results to"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)
