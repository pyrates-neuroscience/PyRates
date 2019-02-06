import os
import pandas as pd
import matplotlib.pyplot as plt

from pyrates.utility import plot_timeseries

# res_dir = "/data/hu_salomon/Documents/ClusterGridSearch/simple_nextgen_NMM/Results/DefaultGrid0/"
res_dir = "/data/hu_salomon/Documents/ClusterGridSearch/CGS_EIC_coupling_test/Results/DefaultGrid0/"

grid_idx = 25
num_header_params = 5
header = list(range(0, num_header_params))
for file in os.listdir(res_dir):
    if file.endswith(f'_idx_{grid_idx}.csv'):
        result = pd.read_csv(f'{res_dir}/{file}', header=header, index_col=0)
        idx = result.columns.tolist()[0]
        ax = plot_timeseries(result[idx[0]][idx[1]])
plt.show()
