from pyrates.utility.cluster_compute import gather_cgs_results
import time as t

print("Start!")

res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example/Results/DefaultGrid_0"

t0 = t.time()
results = gather_cgs_results(res_dir, num_header_params=5)
print(f'Elapsed time: {t.time()-t0:.3f} seconds')


# print(len(results.columns))
# results.to_csv("/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example/Results/test.csv", index=True)
# print(results)