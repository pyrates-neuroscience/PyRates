import pandas as pd


param_grid = pd.read_csv("/data/hu_salomon/Documents/ClusterGridSearch/CGS_nextgen_nmm_test/Grids/DefaultGrid0.csv",
                         index_col=0, header=[0])
param_grid = param_grid.loc[:, param_grid.columns != "status"]

gs_results = pd.read_csv("/data/hu_salomon/Documents/ClusterGridSearch/CGS_nextgen_nmm_test/Results/grid_search_DefaultGrid0_result.csv",
                        index_col=0, header=[0, 1, 2])

cgs_results = "/data/hu_salomon/Documents/ClusterGridSearch/CGS_nextgen_nmm_test/Results/DefaultGrid0"
# print(param_grid)

idx = 0

# for idx, row in param_grid.iterrows():
params = param_grid.iloc[param_grid.index.get_loc(idx), :]
gs_result = gs_results.loc[:, (str(params[0]), str(*params[1:]))]
cgs_result = pd.read_csv(f'{cgs_results}/CGS_result_DefaultGrid0_idx_{idx}.csv', index_col=0, header=[0, 1, 2])

# print(gs_result)
# print(cgs_result)

diff = gs_result.subtract(cgs_result)
print(diff)

