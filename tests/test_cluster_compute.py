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
    import pandas as pd
    import numpy as np

    from pyrates.utility.grid_search import linearize_grid

    # Simple Jansen-Rit circuit
    config_file = f'{tmp_path}/test_config.json'
    subgrid = f'{tmp_path}/test_grid.h5'
    result_file = f'{tmp_path}/test_result.h5'
    build_dir = f'{tmp_path}'

    # Simulation specs
    circuit_template = "model_templates.jansen_rit.simple_jansenrit.JRC"
    dt = 1e-3
    sim_time = 1.0
    param_map = {'u': {'vars': ['RPO_e_pc.0/u'],
                       'nodes': ['PC.0']}}

    # Create param_grid file
    u = np.linspace(100, 400, 10)
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
        "target": [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
    }

    # Create test configuration file for worker template
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Run worker template as console command with arguments
    cmd = ['/data/u_salomon_software/anaconda3/envs/PyRates/bin/python',
           f'/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/worker_template.py',
           f'--config_file={config_file}',
           f'--subgrid={subgrid}',
           f'--local_res_file={result_file}',
           f'--build_dir={build_dir}']
    subprocess.Popen(cmd).wait()

    # Load results from result file
    try:
        results = pd.read_hdf(result_file, key=list(output.keys())[0])
    except FileNotFoundError:
        assert 0

    assert not results.empty


def test_cluster_compute(tmp_path):
    import sys
    import platform
    import pandas as pd
    import numpy as np

    from pyrates.utility.grid_search import linearize_grid, ClusterGridSearch

    #
    nodes = [platform.node()]

    # Simulation specs
    circuit_template = "model_templates.jansen_rit.simple_jansenrit.JRC"
    dt = 1e-3
    sim_time = 1.0
    param_map = {'u': {'vars': ['RPO_e_pc.0/u'],
                       'nodes': ['PC.0']}}

    # Create param_grid file
    u = np.linspace(100, 400, 10)
    param_grid = {'u': u}
    param_grid = linearize_grid(param_grid, permute=False)

    output = {'r': 'PC.0/PRO.0/m_out'}
    cgs = ClusterGridSearch(nodes=nodes, compute_dir=tmp_path)
    res_file = cgs.run(
        circuit_template=circuit_template,
        params=param_grid,
        param_map=param_map,
        simulation_time=sim_time,
        dt=dt,
        sampling_step_size=dt,
        inputs={},
        outputs=output,
        chunk_size=len(u),
        worker_env=sys.executable,
        worker_file='/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/worker_template.py',
        add_template_info=False,
        config_kwargs={
            "init_kwargs": {
                'backend': 'numpy',
                'vectorization': 'nodes',
                'solver': 'euler'
            }
        })

    results = pd.read_hdf(res_file, key=f'Results/{list(output.keys())[0]}')
    assert not results.empty
