from pyrates.cluster_compute.montbrio_eval_funcs import *
from pyrates.cluster_compute.cluster_compute import *


res_file = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/T10s_dt10us/Results/DefaultGrid_2/CGS_result_DefaultGrid_2.h5"
results = pd.read_hdf(res_file, key="/Results/r_E0_df")

with h5py.File(res_file, "r") as file:
    parameters = {
        'k_e': list(file["ParameterGrid/Keys/k_e"]),
        'k_i': list(file["ParameterGrid/Keys/k_i"])
    }

plot_avrg_peak_dist(results, parameters=parameters, tick_size=10)
