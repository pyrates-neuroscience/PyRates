# """Test suite for cluster computation functionality.
# """
#
#
#
# import pytest
#
# # meta infos
# __author__ = "Richard Gast, Daniel Rose"
# __status__ = "Development"
#
# ###########
# # Utility #
# ###########
#
#
# def setup_module():
#     print("\n")
#     print("===================================")
#     print("| Test Suite: Cluster computation |")
#     print("===================================")
#
#
# #########
# # Tests #
# #########
#
# def test_worker_template(tmp_path):
#     """
#
#     Parameters
#     ----------
#     tmp_path
#
#     Returns
#     -------
#
#     """
#
#     import sys
#     import json
#     import subprocess
#     import pandas as pd
#     import numpy as np
#
#     from pyrates.utility.grid_search import grid_search, linearize_grid
#
#     # file paths
#     config_file = f'{tmp_path}/test_config.json'
#     subgrid = f'{tmp_path}/test_grid.h5'
#     result_file = f'{tmp_path}/test_result.h5'
#     build_dir = f'{tmp_path}'
#
#     # Simulation specs
#     circuit_template = "model_templates.jansen_rit.simple_jansenrit.JRC"
#     dt = 1e-3
#     sim_time = 1.0
#     param_map = {'u': {'var': [('RPO_e_pc.0', 'u')],
#                        'nodes': ['PC.0']}}
#
#     # Create param_grid file
#     u = np.linspace(100, 400, 20)
#     param_grid = {'u': u}
#     param_grid = linearize_grid(param_grid, permute=False)
#     param_grid.to_hdf(subgrid, key='subgrid')
#
#     backends = ["numpy"]
#
#     check = []
#
#     for backend in backends:
#         output = {'r': ('PC.0', 'PRO.0', 'm_out')}
#         config_dict = {
#             "circuit_template": circuit_template,
#             "param_map": param_map,
#             "dt": dt,
#             "simulation_time": sim_time,
#             "outputs": output,
#             "backend": backend
#         }
#
#         # Create test configuration file for worker template
#         with open(config_file, "w") as f:
#             json.dump(config_dict, f, indent=2)
#
#         # Run worker computation
#         cmd = [f'{sys.executable}',
#                f'/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/worker_template.py',
#                f'--config_file={config_file}',
#                f'--subgrid={subgrid}',
#                f'--local_res_file={result_file}',
#                f'--build_dir={build_dir}']
#         subprocess.Popen(cmd).wait()
#
#         # Run verification computation
#         results, _t, _ = grid_search(
#             circuit_template=circuit_template,
#             param_grid=param_grid,
#             param_map=param_map,
#             simulation_time=sim_time,
#             dt=dt,
#             permute_grid=False,
#             inputs={},
#             outputs={'r': ('PC.0', 'PRO.0', 'm_out')},
#             init_kwargs={
#                 'backend': backend,
#                 'vectorization': 'nodes'
#             },
#             profile='t',
#             build_dir=build_dir)
#
#         for key in output.keys():
#             worker_results = pd.read_hdf(result_file, key=key)
#             for i, column_name in enumerate(param_grid.values):
#                 worker_result = worker_results.loc[:, tuple(column_name)]
#                 result = results.loc[:, tuple(column_name)]
#                 check.append(worker_result.equals(result))
#
#     assert all(check)
#
#
#
#
#
# # def test_cluster_compute():
# #     pass
# #
#
#
#
# # test worker template with special test files
# # test cgs with only the current workstation as worker