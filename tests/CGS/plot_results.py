import time as t

from pyrates.utility.cluster_grid_search import gather_cgs_results

res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/simple_nextgen_NMM/Results/DefaultGrid_0/"

# filt_list = linearize_grid({'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 8., 2.)}, permute=False)

t0 = t.time()

# result = gather_cgs_results(res_dir, num_header_params=3, filter_=filt_list)
result = gather_cgs_results(res_dir, num_header_params=3)

print('Files loaded. Time elapsed: %.2f seconds' % (t.time()-t0))

print(result)
