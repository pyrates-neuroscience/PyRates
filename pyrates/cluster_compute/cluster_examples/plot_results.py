# internal imports
from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette
import h5py

# PyRates internal imports
from pyrates.cluster_compute.cluster_compute import *


# def create_resultfile(fp_res, fp_h5, delete_temp=False):
#     files = glob.glob(fp_res + "/*.h5")
#     with h5py.File(fp_h5, "w") as f:
#         for i, file_ in enumerate(files):
#             df = pd.read_hdf(file_, key='Data')
#             f.create_dataset(name=f'/Data/Idx_{i}/Values/', data=df.values[:])
#             f.create_dataset(name=f'/Data/Idx_{i}/Columns/', data=df.columns[:])
#             if i == 0:
#                 f.create_dataset(name=f'Index', data=df.index)
#             if delete_temp:
#                 os.remove(file_)


def post(data):
    cut_off = 1.
    _ = plot_psd(data, tmin=cut_off, show=False)
    pow = plt.gca().get_lines()[-1].get_ydata()
    freqs = plt.gca().get_lines()[-1].get_xdata()
    plt.close('all')
    spec = pd.DataFrame(pow, index=freqs)
    spec.columns.names = data.columns.names
    return spec


if __name__ == "__main__":
    res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_Spec/Results/DefaultGrid_0/"
    h5file = f'{res_dir}/CGS_result_DefaultGrid_0.h5'


    # t0 = t.time()
    # create_resultfile(res_dir, h5file)
    # print(f'{t.time()-t0:.3f} seconds')

    results = read_cgs_results(res_dir)
    # data = results.iloc[:,0]
    # print(data.name[-1])
    print(results)
    # print(results.columns.values[-1][-1])
    # data = results.iloc[:, 0].to_frame()
    # cols = data.columns
    # dummy_index = pd.Index(['r_PC.0'])
    # data.columns = dummy_index
    #
    # cut_off = 1.0
    # _ = plot_psd(data, tmin=cut_off, show=False)
    # pow = plt.gca().get_lines()[-1].get_ydata()
    # freqs = plt.gca().get_lines()[-1].get_xdata()
    # plt.close()
    #
    # data = pd.DataFrame(pow, index=freqs, columns=cols)
    #
    # print(data)

    # header_dummy =
    # data.to_frame()
    # # data.columns.names = ['r_PC.0']
    # print(data)






    # temp = data.index.to_frame(index=False)

    # cols = data.columns.to_frame()
    # print(temp)
    # print(type(data.values))
    # print(type(cols))
    # print(cols)
    # print(type(data.columns))
    # vals = data.values
    # idx = data.index
    # cols = data.columns
    #
    # test = pd.DataFrame(vals)
    # test.set_index(idx)
    # test.columns = cols
    #
    # print(test)

    # data_1 = pd.read_hdf(h5file, key="Idx_0/Data")
    # print(data_1)
    #
    # data_2 = pd.read_hdf(h5file, key="Idx_1/Data")
    # print(data_2)
    # with h5py.File(h5file, "r") as f:
    #     data = list(f["0/Data"])
    #     data_new = pd.DataFrame(data)
    # print(data_new)


    # if filter_grid:
    #     filter_grid = filter_grid.values.tolist()

    # list_ = []
    # for file_ in files:
    #     try:
    #         df = pd.read_hdf(file_, key='Data')
    #         list_.append(df)
    #     finally:
    #         pass




    # grid_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example/Grids/DefaultGrid_0.h5"
    # param_grid = pd.read_hdf(grid_dir, key='Data')
    # results = read_cgs_results(res_dir)

    # with h5py.File(h5file, "w") as f:
    #     for i, col in enumerate(results.columns):
    #         temp = (results.iloc[:, 0]).to_frame()
    #         temp.columns.names = results.columns.names
    #         dset = f.create_dataset(f'{i}/Data', data=temp)

    # with h5py.File(h5file, "r") as f:
    #     data = list(f["0/Data"])
    #     data_new = pd.DataFrame(data)
    # print(data_new)

            # # result_temp = results.iloc[:, col]
            # result_temp = results[10][6]
            # idx_label = result_temp.name[:-1]
            # idx = param_grid[(param_grid.values == idx_label).all(1)].index
            # # result = result_temp.to_frame()
            # result = result_temp







    # # print(type(results.iloc[:,0]))
    # temp = (results.iloc[:,0]).to_frame()
    # temp.columns.names = results.columns.names
    # print(type(temp))
    # print(temp)
    # print(type(results[10][6]))
    # print(results[10][6])


    # for col in range(len(results.columns)):
    #     # result_temp = results.iloc[:, col]
    #     result_temp = results[10][6]
    #     idx_label = result_temp.name[:-1]
    #     idx = param_grid[(param_grid.values == idx_label).all(1)].index
    #     # result = result_temp.to_frame()
    #     result = result_temp
    #
    #     result = post(result_temp)
    #
    #     result.columns.names = results.columns.names
    #
    #     print(result)
    # fig, ax = plt.subplots(ncols=2, figsize=(15, 5), gridspec_kw={})
    # plt.show()

    # cm1 = cubehelix_palette(n_colors=int(len(params['H_e']) * len(params['H_i'])), as_cmap=True, start=2.5, rot=-0.1)
    # cm2 = cubehelix_palette(n_colors=int(len(params['H_e']) * len(params['H_i'])), as_cmap=True, start=-2.0, rot=-0.1)
    # cax1 = plot_connectivity(max_freq, ax=ax[0], yticklabels=list(np.round(params['H_e'], decimals=2)),
    #                          xticklabels=list(np.round(params['H_i'], decimals=2)), cmap=cm1)
    # cax1.set_xlabel('H_i')
    # cax1.set_ylabel('H_e')
    # cax1.set_title(f'max freq')
    # cax2 = plot_connectivity(freq_pow, ax=ax[1], yticklabels=list(np.round(params['H_e'], decimals=2)),
    #                          xticklabels=list(np.round(params['H_i'], decimals=2)), cmap=cm2)
    # cax2.set_xlabel('H_i')
    # cax2.set_ylabel('H_e')
    # cax2.set_title(f'freq pow')
    # plt.suptitle('EI-circuit sensitivity to synaptic efficacies (H)')
    # plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))
    # # fig.savefig("/home/rgast/Documents/Studium/PhD_Leipzig/Figures/BGTCS/eic_efficacies", format="svg")
    # plt.show()


    # print(pow)




# res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example_3/Results/DefaultGrid_2/"
# results = read_cgs_results(res_dir)
# print(results)
# print(results.columns.values)
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