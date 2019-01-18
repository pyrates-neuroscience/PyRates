# # system imports
import argparse
import sys
import ast
import json

# external imports
import pandas as pd
# from numpy import array
#
# from pyrates.utility import grid_search



def main(_):
    with open(FLAGS.config_file) as file:
        param_dict = json.load(file)
        try:
            circuit_template = param_dict['circuit_template']
            param_map = param_dict['param_map']
            # Recreate tuple from string representation as 'key' in inputs
            inputs = {ast.literal_eval(*param_dict['inputs'].keys()):
                      list(*param_dict['inputs'].values())}
            # Recreate tuple from list as 'values' in outputs
            outputs = {str(*param_dict['outputs'].keys()):
                       tuple(*param_dict['outputs'].values())}
            sampling_step_size = param_dict['sampling_step_size']
            dt = param_dict['dt']
            simulation_time = param_dict['simulation_time']
        except KeyError as err:
            # If config_file does not contain a key named 'param_grid':
            print("KeyError:", err)
            return

    # Recreate dict param_grid from its string representation and create a DataFrame from it
    param_grid = pd.DataFrame(ast.literal_eval(FLAGS.param_grid_arg))

    # TODO: Await a param_grid from stdin to start grid_search()
    # TODO: Start grid_search() and print results
    # print("Computing something")
    x = 0
    for i in range(200000000):
        x = x + 1
    # print("Computation finished")





    # print(f'circuit template: {type(circuit_template)}')
    # # print(f'param_grid: {type(param_grid)}')
    # print(f'param_map: {type(param_map)}')
    # print(f'inputs: {type(inputs)}')
    # print(f'outputs: {type(outputs)}')
    # print(f'sampling_step_size: {type(sampling_step_size)}')
    # print(f'dt: {type(dt)}')
    # print(f'simulation_time: {type(simulation_time)}')

    # Exclude 'status'-key param_grid because grid_search() can't handle the additional keyword
    # param_grid_arg = param_grid.loc[:, param_grid.columns != "status"]
    #
    # results = grid_search(circuit_template=circuit_template,
    #                       param_grid=param_grid_arg,
    #                       param_map=param_map,
    #                       inputs=inputs,
    #                       outputs=outputs,
    #                       sampling_step_size=sampling_step_size,
    #                       dt=dt,
    #                       simulation_time=simulation_time)
    #
    # print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--param_grid_arg",
        type=str,
        default="",
        help="Parameter grid to use grid_search() on"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="JSON file containing all necessary input to invoke grid_search"
    )

    FLAGS = parser.parse_args()

    main(sys.argv)