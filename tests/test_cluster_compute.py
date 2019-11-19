# """Test suite for cluster computation functionality.
# """
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
#
# def test_1_cluster_compute(tmp_path):
#     import platform
#     import pandas as pd
#     import numpy as np
#
#     from pyrates.utility.grid_search import ClusterGridSearch
#
#     nodes = [platform.node()]
#
#     # Simulation specs
#     circuit_template = "model_templates.jansen_rit.simple_jansenrit.JRC"
#     dt = 1e-3
#     sim_time = 1.0
#     param_map = {'u': {'vars': ['RPO_e_pc/u'],
#                        'nodes': ['PC']}}
#
#     # Create param_grid file
#     u = np.linspace(100, 400, 10)
#     param_grid = {'u': u}
#
#     output = {'r': 'PC/PRO/m_out'}
#     cgs = ClusterGridSearch(nodes=nodes, compute_dir=tmp_path)
#     res_file = cgs.run(
#         circuit_template=circuit_template,
#         param_grid=param_grid,
#         param_map=param_map,
#         simulation_time=sim_time,
#         dt=dt,
#         permute_grid=False,
#         sampling_step_size=dt,
#         inputs={},
#         outputs=output,
#         chunk_size=len(u),
#         add_template_info=False,
#         verbose=True,
#         gs_kwargs={
#             "init_kwargs": {
#                 'backend': 'numpy',
#                 'solver': 'euler'
#             },
#             "njit": True,
#             "parallel": False
#         },
#         worker_kwargs={"test": "test"}
#     )
#
#     try:
#         results = pd.read_hdf(res_file, key=f'Results/results')
#     except KeyError:
#         assert False
#     assert not results.empty
#
