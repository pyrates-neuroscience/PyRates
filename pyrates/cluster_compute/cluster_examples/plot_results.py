# External imports
import matplotlib.pyplot as plt

# PyRates internal imports
from pyrates.utility import plot_timeseries
from pyrates.cluster_compute.cluster_compute import *

res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example_3/Results/DefaultGrid_2/"
results = gather_cgs_results(res_dir)
print(results)
print(results.columns.values)
# results.to_hdf("/data/hu_salomon/Documents/test.h5", key="Data")
# print(results)
# print(results.columns.values)
# file = "/data/hu_salomon/Documents/test.h5"
# results = pd.read_hdf(file, key='Data')
# print(results)
# print(results.columns.values)
# print(results[7.0][8.0])
# fp = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_2/Results/DefaultGrid_2/CGS_result_DefaultGrid_2.h5"
#
# f = h5py.File(fp)
# print(list(f))
# list_ = list(f)
# f.close()
#
# print(list_)
# print(len(list_))
#
# res = []
# for i in list_:
#     res.append(pd.read_hdf(fp, key=i))
# results = pd.concat(res, axis=1)
# print(results)

# print(list(f))
# print(f['df'])

# df = pd.read_hdf(fp, key='2')
# print(df)
# print(df.columns.values)

# res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example_3/Results/DefaultGrid_0/"
#
# t0 = t.time()
# results = gather_cgs_results(res_dir, num_header_params=5)
# print(f'Elapsed time: {t.time()-t0:.3f} seconds')
#
# print(results)
# print(results.columns.values)

# params = pd.read_csv("/data/hu_salomon/Documents/testgrid_2.csv",
#                      index_col=0)
#
# print(params)
# print(type(params.iloc[4,0]))

# print(results)
# print(type(results.iloc[10,0]))
# print(results.columns.values)
# params = pd.read_csv("/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_2/Grids/DefaultGrid_0.csv",
#                      index_col=0, header=0)
########
# EVAL #
########
# print(results.loc[:, (params['J_e'][0], params['J_i'][0])])

# plotting
# for j_e in params['J_e']:
#     for j_i in params['J_i']:
#         print(j_e, j_i)
#         ax = plot_timeseries(results[str(j_e)][str(j_i)], title=f"J_e={j_e}, J_i={j_i}")
# plt.show()

# print(len(results.columns))
# results.to_csv("/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example/Results/test.csv", index=True)
# print(results)