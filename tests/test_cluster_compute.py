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

    import json
    import subprocess
    import numpy as np

    from pyrates.utility.grid_search import grid_search, linearize_grid

    # Simple Jansen-Rit circuit
    test_path = "data/hu_salomon/Documents/CGSWorkerTests/simple_test_model"
    config_file = f'{tmp_path}/test_config.json'
    subgrid = f'{tmp_path}/test_grid.h5'
    result_file = f'{tmp_path}/test_result.h5'
    build_dir = f'{tmp_path}'

    # Simulation specs
    circuit_template = "model_templates.jansen_rit.simple_jansenrit.JRC"
    dt = 1e-3
    sim_time = 1.0
    param_map = {'u': {'var': [('RPO_e_pc.0', 'u')],
                       'nodes': ['PC.0']}}

    # Create param_grid file
    u = np.linspace(100, 400, 20)
    param_grid = {'u': u}
    param_grid = linearize_grid(param_grid, permute=False)
    param_grid.to_hdf(subgrid, key='subgrid')

    output = {'r': 'PC.0/PRO.0/m_out'}
    config_dict = {
        "circuit_template": circuit_template,
        "param_map": param_map,
        "dt": dt,
        "simulation_time": sim_time,
        "outputs": output,
        "backend": 'numpy',
        "init_kwargs": {
            'backend': 'numpy',
            'vectorization': 'nodes',
            'solver': 'euler'
        },
    }

    # Create test configuration file for worker template
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    cmd = ['/data/u_salomon_software/anaconda3/envs/PyRates/bin/python',
           f'/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/worker_template.py',
           f'--config_file={config_file}',
           f'--subgrid={subgrid}',
           f'--local_res_file={result_file}',
           f'--build_dir={build_dir}']
    subprocess.Popen(cmd).wait()

    # results_1 = pd.read_hdf(result_file, key='r')
    # results_2, _t, _ = grid_search(
    #     circuit_template=circuit_template,
    #     param_grid=param_grid,
    #     param_map=param_map,
    #     simulation_time=sim_time,
    #     dt=dt,
    #     permute_grid=False,
    #     inputs={},
    #     outputs={'r': ('PC.0', 'PRO.0', 'm_out')},
    #     init_kwargs={
    #         'backend': 'numpy',
    #         'vectorization': 'nodes'
    #     },
    #     profile='t')
    # # #
    # res_lst = []
    # for i, column_values in enumerate(param_grid.values):
    #     result_3 = results_2.loc[:, column_values]
    #     result_3.columns.names = results_2.columns.names
    #     res_lst.append(result_3)
    # results_3 = pd.concat(res_lst, axis=1)
    #
    # print(results_1 == results_2)
    #
    # check = []
    # for i, column_name in enumerate(param_grid.values):
    #     result_1 = results_1.loc[:, column_name]
    #     result_2 = results_2.loc[:, column_name]
    #     result_3 = results_3.loc[:, column_name]
    #     check.append((result_1 == result_2).all()[0])
    # print(check)

    # test against explicit graph computation
    assert 0




# def test_cluster_compute():
#     pass
#



# test worker template with special test files
# test cgs with only the current workstation as worker