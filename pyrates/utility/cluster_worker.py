# # system imports
import argparse
import sys
import ast
import json

# external imports
import pandas as pd
from pyrates.utility import grid_search


def dummy():
    x = 0
    for i in range(200000000):
        x = x + 1


def main(_):
    with open(FLAGS.config_file) as file:
        param_dict = json.load(file)
        try:
            circuit_template = param_dict['circuit_template']
            param_map = param_dict['param_map']

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

    # TODO: Send output_file back to host if there is no shared memory space available


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--param_grid_arg",
        type=str,
        default="",
        help="String representation of parameter grid chunk for grid_search()"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="JSON file containing all necessary input to invoke grid_search()"
    )

    parser.add_argument(
        "--result_path",
        type=str,
        default="",
        help="Path to save result files to"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)
