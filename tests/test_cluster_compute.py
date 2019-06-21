"""Test suite for cluster computation functionality.
"""



import pytest

# meta infos
__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

###########
# Utility #
###########


def setup_module():
    print("\n")
    print("===================================")
    print("| Test Suite: Cluster computation |")
    print("===================================")


#########
# Tests #
#########

def test_worker_template(tmp_path):
    """

    Parameters
    ----------
    tmp_path

    Returns
    -------

    """

    import os
    import json
    import subprocess
    import pandas as pd

    from pyrates.utility.grid_search import linearize_grid

    # file paths
    config_file = f'{tmp_path}/test_config.json'
    subgrid = f'{tmp_path}/test_grid.h5'
    result_file = f'{tmp_path}/test_result.h5'
    # result_file = f'/data/hu_salomon/Documents/CGSWorkerTests/test/test_result.h5'

    # Simulation specs
    circuit_template = "model_templates.test_resources.test_compute_graph.net5"
    dt = 1e-1
    sim_time = 10.0
    param_map = {'u': {'var': [('op5.0', 'u')],
                       'nodes': ['pop0.0']}}

    # Create param_grid file
    u = list(range(10))
    param_grid = {'u': u}
    param_grid = linearize_grid(param_grid, permute=False)
    param_grid.to_hdf(subgrid, key='subgrid')

    backends = ["numpy"]
    # backends = ["numpy", "tensorflow"]

    for backend in backends:
        output = {'a': ('pop0.0', 'op5.0', 'a')}
        config_dict = {
            "circuit_template": circuit_template,
            "param_map": param_map,
            "dt": dt,
            "simulation_time": sim_time,
            "outputs": output,
            "backend": backend
        }

        # Create test configuration file for worker template
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

        cmd = ['/data/u_salomon_software/anaconda3/envs/PyRates/bin/python',
               f'{os.getcwd()}/../pyrates/utility/worker_template.py',
               f'--config_file={config_file}',
               f'--subgrid={subgrid}',
               f'--local_res_file={result_file}']
        # subprocess.Popen(cmd).wait()
        subprocess.Popen(f'/data/u_salomon_software/anaconda3/envs/PyRates/bin/python' + f'{os.getcwd()}/../pyrates/utility/worker_template.py --config_file={config_file} --subgrid={subgrid} --local_res_file={result_file}', shell=True).wait()


        # worker_results = pd.read_hdf(result_file, key="a")

    # test against explicit graph computation
    assert 0




# def test_cluster_compute():
#     pass
#



# test worker template with special test files
# test cgs with only the current workstation as worker