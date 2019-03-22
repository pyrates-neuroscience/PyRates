# internal imports
from collections import OrderedDict
import time
import pandas as pd
from pyrates.utility import plot_timeseries, grid_search, plot_psd, plot_connectivity
from pyrates.utility.grid_search import linearize_grid
from pyrates.cluster_compute.cluster_compute import read_cgs_results

from seaborn import cubehelix_palette
import h5py
import scipy.signal as sp
import seaborn as sb
# PyRates internal imports
from seaborn import set
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


def post(data_):
    cut_off = 1.
    _ = plot_psd(data_, tmin=cut_off, show=False)
    pow = plt.gca().get_lines()[-1].get_ydata()
    freqs = plt.gca().get_lines()[-1].get_xdata()
    plt.close('all')
    spec = pd.DataFrame(pow, index=freqs)
    spec.columns.names = data_.columns.names
    return spec


if __name__ == "__main__":
    import glob

    fp = "/nobackup/spanien1/salomon/ClusterGridSearch/TestData/Test_result.h5"
    data = pd.read_hdf(fp, key='Data')
    print(data.iloc[:,40])
    # with h5py.File(fp, "r") as file:
    #     test = file["invalid_key"]
    # test = pd.read_hdf(fp, key="invalid_key")

    # for file in glob.glob(f'{fp}/*_temp_*.h5'):
    #     print(file)



    # # results = pd.read_hdf('/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/Test/Results/DefaultGrid_3/CGS_result_DefaultGrid_3_idx_0-40_temp.h5')
    # # print(results)
    #
    # params = OrderedDict()
    # params['k_e'] = np.linspace(20., 30., 101)
    # params['k_i'] = np.linspace(10., 20., 101)
    #
    # param_grid = linearize_grid(params, permute=True)
    # results = pd.read_hdf('/data/hu_salomon/Documents/test.h5')
    #
    # print(results.columns.values[0][-1])
    #
    # temp = []
    # for i in range(5):
    #     idx_label = param_grid.iloc[i].values.tolist()
    #     idx_label.append('r_E.0')
    #
    #     result = results.loc[:, tuple(idx_label)].to_frame()
    #     result.columns.names = results.columns.names
    #     temp.append(result)
    # frame = pd.concat(temp, axis=1)
    # print(frame)

    # slice_ = results.columns.get_loc((20, 10))
    # print(results.columns.levels[slice])
    # slice_ = results.get_loc(tuple(idx_label))
    # mi = pd.MultiIndex.from_list(idx_label)
    # mi.levels()

    # result.columns = tuple(idx_label)

    # result.columns.values = pd.Index(list(idx_label))
    # print(result)
    #
    # slice_ = results.get_loc((20, 10))
    # print(results[slice_])
    #
    #
    #
    # temp = []
    # for i in range(5):
    #     idx_label = tuple(param_grid.iloc[i].values)
    #
    #     result = results.loc[:, idx_label]
    #     res_idx =
    #     temp = results[(results.columns.names[:-1] in idx_label).all(1)]
    #     print(temp)
    #     temp.append(results.loc[:, idx_label])
    # frame = pd.concat(temp, axis=1)
    # print(frame)


    # list_ = [results.iloc[:,0] * 5]
    # temp = results.copy(deep=True)
    #
    # print(results)
    #
    # for col in range(5):
    #     result = results.iloc[:, col]
    #     idx_label = result.name[:-1]
    #     idx = param_grid[(param_grid.values == idx_label).all(1)].index
    #     temp.iloc[:,idx+1] = result.to_frame()
    #
    # print(temp)
    # frame = pd.concat(list_, axis=1)

    # print(type(results[params['k_e'][0]][params['k_i'][0]]))
    # list_ = []
    # print(results.columns.values[:][:-1])
    #
    # for i in range(5):
    #     idx = param_grid.iloc[i,:]
    #     # temp = results[(results.columns.names[:-1] == idx.values).all(1)]
    #     temp = results.loc[:,idx.all(1)]
    #     print(temp)
    #
    # for i in range(5):
    #     temp = results.iloc[:, i]
    #     list_.append(temp)
    # frame = pd.concat(list_, axis=1)
    # frame.columns.names = results.columns.names
    # print(frame)
    # print(type(frame))

    #
    #
    #
    #
    # pass
    # params = {'k_e': np.linspace(20., 30., 101), 'k_i': np.linspace(10., 20., 101)}
    # param_grid = linearize_grid(params, permute=True)
    # for i, pair in enumerate(param_grid.values):
    #     print(pair[0], pair[1])

    # fp = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio_EIC/Computation_1/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5"
    # results = read_cgs_results(fp, key='Num_Peaks')
    #
    # # print((np.array(results[29.0][14.0]).astype('float')))
    # print(results[29.0][14.0].values.item())
    #
    # #
    # params = {'k_e': np.linspace(20., 30., 101), 'k_i': np.linspace(10., 20., 101)}
    # param_grid = linearize_grid(params, permute=True)
    # print(len(param_grid.index))
    # print(type(param_grid.values))
    # T = 2.0
    #
    # param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
    #                      'nodes': ['E.0', 'I.0']},
    #              'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
    #                      'nodes': ['E.0', 'I.0']}
    #              }
    # params = {'k_e': np.linspace(15., 20., 11), 'k_i': np.linspace(15., 20., 11)}
    # circuit_template = "/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
    # dt = 1e-5,
    # inputs = {},
    # outputs = {"r": ("E", "Op_e.0", "r")},
    # sampling_step_size = 1e-3
    #
    # param_grid = linearize_grid(params, permute=True)
    # test_file = '/data/hu_salomon/Documents/test.h5'
    #
    # # print(param_grid.values)
    #
    # with h5py.File(test_file, 'w') as file:
    #     for key, value in params.items():
    #         file.create_dataset(f'ParameterGrid/{key}', data=value)
    #     file.create_dataset(f'ParameterGrid/Grid', data=param_grid.values)



    # file = '/nobackup/spanien1/salomon/ClusterGridSearch/EIC_Montbrio/Computation_1/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5'
    # data = read_cgs_results(file, key='Data')
    # print(data)



    # res_file = '/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_spike_rate/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5'
    # # results = read_cgs_results(res_file, key='Data/Idx_5712/Data')
    # results = pd.read_hdf(res_file, key='GridIndex/Idx_5712/Data')
    # data = results.iloc[0:int((1/1e-3)), :]
    # cm = cubehelix_palette(n_colors=1, as_cmap=False, start=6, rot=-0.1)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    # plot_timeseries(data, ax=ax, cmap=cm)
    # plt.tick_params(labelsize=20)
    # axis_font = {'fontname': 'Arial', 'size': '25'}
    # ax.set_xlabel('t in s', fontdict=axis_font)
    # ax.set_title('k_e = 21.2, k_i = 21.2', fontdict=axis_font)
    # fig.savefig("/data/hu_salomon/Documents/MA/Graphics/Plots/EIC_spike_Helmut", format="svg")
    # plt.show()


    # params = {'J_e': np.linspace(10., 30., 101), 'J_i': np.linspace(10., 30., 101)}
    # num_peaks = np.zeros([len(params['J_e']), len(params['J_i'])])
    # T = 4.0
    # for m, j_e in enumerate(params['J_e']):
    #     for n, j_i in enumerate(params['J_i']):
    #         num_peaks[m, n] = np.array(results[j_e][j_i])
    #         # data = np.array(results[j_e][j_i])
    #         # peaks = sp.argrelextrema(data, np.greater)
    #         # num_peaks[m, n] = int(len(peaks[0]) / T)
    #         # plot_timeseries(results[j_e][j_i])
    # # num_peaks_temp = read_cgs_results(res_file, key='Num_Peaks')
    # # num_peaks = num_peaks_temp.values.reshape(len(params['J_e']), len(params['J_i']))
    #
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    # plot_connectivity(num_peaks, ax=ax, xticklabels=list(params['J_e']), yticklabels=list(params['J_i']))
    # for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if n % 25 != 0:
    #         label.set_visible(False)
    # for n, label in enumerate(ax.yaxis.get_ticklabels()):
    #     label.font_size = 20
    #     if n % 25 != 0:
    #         label.set_visible(False)
    # plt.tick_params(labelsize=20)
    # axis_font = {'fontname': 'Arial', 'size': '25'}
    # ax.set_xlabel('k_e', fontdict=axis_font)
    # ax.set_ylabel('k_i', fontdict=axis_font)
    # ax.set_title('Average spikes per second', fontdict=axis_font)
    #
    # fig.savefig("/nobackup/spanien1/salomon/EIC_spike_detection/CGS_EIC_spike_detection", format="svg")
    #
    # plt.show()

    # num_peaks = np.zeros([len(params['J_e']), len(params['J_i'])])
    # T = 4.0
    # for m, j_e in enumerate(params['J_e']):
    #     for n, j_i in enumerate(params['J_i']):
    #         data = np.array(results[j_e][j_i])
    #         peaks = sp.argrelextrema(data, np.greater)
    #         num_peaks[m, n] = int(len(peaks[0]) / T)
    #         # plot_timeseries(results[j_e][j_i])
    #
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    # h = plot_connectivity(num_peaks, ax=ax, xticklabels=list(params['J_e']), yticklabels=list(params['J_i']))
    # for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if n % 5 != 0:
    #         label.set_visible(False)
    # for n, label in enumerate(ax.yaxis.get_ticklabels()):
    #     if n % 5 != 0:
    #         label.set_visible(False)
    # ax.set_xlabel('k_e')
    # ax.set_ylabel('k_i')
    # ax.set_title('Average number of spikes per second')
    # sb.set(font_scale=4)
    # # fig.savefig("/data/hu_salomon/Documents/EIC_spike_detection/EIC_spike_detection", format="svg")
    # plt.show()

    #
    #
    #
    #
    # spec_results = read_cgs_results(res_file, key='Spec')
    # data_temp = spec_results.loc['max_freq', :].to_frame()
    # vals = data_temp.values.reshape(len(params['J_e']), len(params['J_i']))
    # # print(data_temp.values)
    # data = pd.DataFrame(vals, index=data_temp.index.levels[0], columns=data_temp.index.levels[1])
    # print(data)
    # result_map.index = params['J_e']
    # result_map.columns.values = params['J_i']
    # print(result_map)
    #
    # plot_connectivity(data)
    # #
    # # print(f'Data loaded. Elapsed time: {t.time() - t0} seconds')
    # #
    # plt.show()

    # list_ = []
    # with(h5py.File(res_file, 'r')) as file_:
    #     keys = list(file_['GridIndex'].keys())
    #
    #
    #
    # with pd.HDFStore(res_file, "r") as store:
    #     for i, index_key in enumerate(keys):
    #         df = store.get(key=f'/GridIndex/{index_key}/Data/')
    #         list_.append(df)
    #
    # results = pd.concat(list_, axis=1)
    # # print(results)
    #
    # print(f'Elapsed time: {t.time()-t0:.3f} seconds')

    #
    # fp_res = '/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_test/Results/DefaultGrid_14'
    # # fp_s1 = '/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_test/Results/DefaultGrid_14/CGS_result_DefaultGrid_14_idx_50-99.h5'
    # # fp_s2 = '/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_test/Results/DefaultGrid_14/CGS_result_DefaultGrid_14_idx_0-49.h5'
    # fp_d =  '/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_test/Results/DefaultGrid_14/CGS_result_DefaultGrid_14.h5'
    # with h5py.File(fp_d, 'w') as fd:
    #
    #     for file in glob.glob(fp_res + "/*.h5"):
    #         if 'GridIndex/' not in fd.keys():
    #             group = fd.create_group(f'GridIndex/')
    #         else:
    #             group = fd['GridIndex']
    #         if file != fp_d:
    #             fs = h5py.File(file, 'r')
    #             for key in list(fs['GridIndex'].keys()):
    #                 fs.copy(f'GridIndex/{key}', group)
    #             fs.close()

        # with h5py.File(fp_s1, 'r') as fs:
        #     for key in list(fs['GridIndex'].keys()):
        #         group = fd.create_group(f'GridIndex/{key}')
        #         fs.copy(f'GridIndex/{key}/Data', group)
        # with h5py.File(fp_s2, 'r') as fs:
        #     for key in list(fs['GridIndex'].keys()):
        #         group = fd.create_group(f'GridIndex/{key}')
        #         fs.copy(f'GridIndex/{key}/Data', group)
        # with h5py.File(fp_s2, 'r') as fs:
        #     fs.copy('GridIndex', fd)

            # data = fs['GridIndex/60/Data/block0_values']
            # print(data)
            # for key in list(group.keys()):
            #     data = fs[f'/GridIndex/{key}']
            #     fd.create_dataset(name=f'GridIndex/{key}', data=data)
            # print(group.keys())
            # fd.
        # print(list(f.keys()))




    # res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_New/Results/DefaultGrid_12/"
    # h5file = f'{res_dir}/CGS_result_DefaultGrid_0.h5'
    #
    # res_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_EIC_coupling_example_2/Results/DefaultGrid_0/"
    # spec = read_cgs_results(res_dir, key='Spec')
    #
    # cm = cubehelix_palette(n_colors=int(spec.size), as_cmap=True, start=-2.0, rot=-0.1)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    # spec.index = np.round(spec.index, decimals=2)
    # cax = plot_connectivity(spec, ax=ax, yticklabels=10,
    #                         xticklabels=False, cmap=cm)
    # plt.yticks(fontsize=15, rotation='horizontal')
    # # for tick in cax.yaxis.get_major_ticks():
    # #     tick.label.set_fontsize(15)
    # #     tick.label.set_rotation('horizontal')
    # cax.set_xlabel('test', size=20)
    # cax.set_ylabel('test', size=20)
    # cax.set_title('test', size=20)

    # time_results = read_cgs_results(res_dir, key='Data')
    # spec_results = read_cgs_results(res_dir, key='Spec')
    #
    # x = 10
    # t = pd.DataFrame([x])
    # f = t.values
    # print(f)

    # t = pd.DataFrame(data=x)
    # print(t)

    # time_results = read_cgs_results(res_dir, key='Data')
    # spec_results = read_cgs_results(res_dir, key='Spec')

    # cm2 = cubehelix_palette(n_colors=int(spec_results.size), as_cmap=True, start=2.5, rot=-0.1)
    #
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # ax1.imshow(time_results.values)
    # ax2.imshow(spec_results.values, cmap=cm2)
    #
    # fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(20, 15), gridspec_kw={})
    # cax1 = plot_connectivity(time_results.values, ax=ax[0], yticklabels=list(time_results.index),
    #                          xticklabels=np.arange(len(time_results.columns.values)), cmap=cm1)
    # # cax2 = plot_connectivity(spec_results.values, ax=ax[1], yticklabels=list(spec_results.index),
    # #                          xticklabels=np.arange(len(spec_results.columns.values)), cmap=cm1)
    # plt.show()




    #

    # t0 = t.time()
    # create_resultfile(res_dir, h5file)
    # print(f'{t.time()-t0:.3f} seconds')

    # results = read_cgs_results(res_dir, key='Spec')
    # data = results.values
    #
    # print(type(results.values))
    # print(results.columns)
    # print(len(results.columns.values))


    #
    # res = results.values
    # # print(res)
    #
    # fig, ax = plt.subplots()
    # cm1 = cubehelix_palette(n_colors=int(results.size), as_cmap=True, start=-2.0, rot=-0.1)
    # plt.imshow(results.values, cmap=cm1)
    # plt.show()
    #
    #
    # np.arrange()
    # cm1 = cubehelix_palette(n_colors=int(results.size), as_cmap=True, start=2.5, rot=-0.1)
    # ax.hist2d(x=len(results.index), y=len(results.columns.values), results.values)
    # plt.show()
    # cax1 = plot_connectivity(res, ax=ax, cmap=cm1)
    # cax1.set_xlabel('intra/inter pcs')
    # cax1.set_ylabel('exc/inh pcs')
    # cax1.set_title(f'max freq')



    # data = results.iloc[:,0]
    # print(data.name[-1])
    # print(results)
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