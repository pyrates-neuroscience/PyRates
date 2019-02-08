# import os
import glob
import pandas as pd
# import matplotlib.pyplot as plt
#
# from pyrates.utility import plot_timeseries
#
from pyrates.utility.cluster_grid_search import gather_cgs_results

# res_dir = "/data/hu_salomon/Documents/ClusterGridSearch/simple_nextgen_NMM/Results/DefaultGrid0/"
print("Start")

res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_nmm_test2/Results/DefaultGrid_1/"

results = gather_cgs_results(res_dir, params=3)
print(results)

# files = glob.glob(res_dir + "/*.csv")
#
# list_ = []
# for file_ in files:
#     df = pd.read_csv(file_, index_col=0, header=[0,1,2])
#     list_.append(df)
#
# # print(list_)
# results = pd.concat(list_, axis=1)
# results.to_csv("/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_nmm_test2/Results/test.csv", index=True)
# print(results)

# grid_idx = 25
# num_header_params = 5
# header = list(range(0, num_header_params))
# for file in os.listdir(res_dir):
#     if file.endswith(f'_idx_{grid_idx}.csv'):
#         result = pd.read_csv(f'{res_dir}/{file}', header=header, index_col=0)
#         idx = result.columns.tolist()[0]
#         ax = plot_timeseries(result[idx[0]][idx[1]])
# plt.show()
