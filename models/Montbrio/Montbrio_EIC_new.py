# Internal imports
import time as t

# External imports
import scipy.signal as sp
import numpy as np


# PyRates internal imports
from pyrates.utility import grid_search


t0 = t.time()
param_map = {'k_ee': {'var': [('Op_e.0', 'k_ee')],
                      'nodes': ['E.0']},
             'k_ei': {'var': [('Op_e.0', 'k_ei')],
                      'nodes': ['E.0']},
             'k_ii': {'var': [('Op_i.0', 'k_ii')],
                      'nodes': ['I.0']},
             'k_ie': {'var': [('Op_i.0', 'k_ie')],
                      'nodes': ['I.0']}
             }

Cs = [15.0, 20.0, 25.0, 30.0]

# ei_ratio = k_ee/k_ii
# io_ratio = k_ee/k_ie
ei_ratio = np.linspace(0.5, 2.0, 4)
io_ratio = np.linspace(0.5, 4.0, 4)

###############
# Grid Search #
###############
t0 = t.time()
dts = 1e-2

fig, ax = plt.subplots(ncols=len(Cs), nrows=2, figsize=(20, 15), gridspec_kw={})
for idx, C in enumerate(Cs):
    k_ee = np.zeros((int(len(ei_ratio) * len(io_ratio))))
    k_ei = np.zeros_like(k_ee)
    k_ie = np.zeros_like(k_ee)
    k_ii = np.zeros_like(k_ee)

    n = 0
    k_ee += C
    for ei in ei_ratio:
        for io in io_ratio:
            k_ei[n] += C / (ei * io)
            k_ie[n] += C / io
            k_ii[n] += C / ei
            n += 1

    params = {'k_ee': k_ee, 'k_ei': k_ei, 'k_ie': k_ie, 'k_ii': k_ii}

    results = grid_search(circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                          param_grid=params,
                          param_map=param_map,
                          dt=5e-4,
                          simulation_time=150.0,
                          inputs={},
                          outputs={"r": ("E", "Op_e.0", "r")},
                          sampling_step_size=dts,
                          permute_grid=False)

    print(f'Grid search elapsed time: {t.time()-t0} seconds')

    peaks_freq = np.zeros((len(ei_ratio), len(io_ratio)))
    peaks_amp = np.zeros_like(peaks_freq)
    for k_ee_, k_ei_, k_ie_, k_ii_ in zip(params['k_ee'], params['k_ei'], params['k_ie'], params['k_ii']):
        if not results[k_ee_][k_ei_][k_ie_][k_ii_].isnull().any().any():
            r, c = np.argmin(np.abs(ei_ratio - k_ee_ / k_ii_)), np.argmin(np.abs(io_ratio - k_ee_ / k_ie_))
            data = np.array(results[k_ee_][k_ei_][k_ie_][k_ii_].loc[50.0:])
            peaks, props = sp.find_peaks(data.squeeze(), prominence=0.6*(np.max(data) - np.mean(data)), distance=1/dts)
            if len(peaks) > 1:
                diff = np.mean(np.diff(peaks)) * dts * 0.01
                peaks_freq[r, c] = 1 / diff
                peaks_amp[r, c] = np.mean(props['prominences'])

    cm1 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=2.5, rot=-0.1)
    cax1 = plot_connectivity(peaks_freq, ax=ax[0, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm1)
    cax1.set_xlabel('intra/inter pcs')
    cax1.set_ylabel('exc/inh pcs')
    cax1.set_title(f'max freq (C = {C})')

    cm2 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
    cax2 = plot_connectivity(peaks_amp, ax=ax[1, idx], yticklabels=list(np.round(ei_ratio, decimals=2)),
                             xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm2)
    cax2.set_xlabel('intra/inter pcs')
    cax2.set_ylabel('exc/inh pcs')
    cax2.set_title(f'mean peak amp (C = {C})')

plt.suptitle('EI-circuit sensitivity to population Coupling strengths (pcs)')
# plt.tight_layout(pad=2.5, rect=(0.01, 0.01, 0.99, 0.96))

plt.show()

    #
    # t_run = t.time()-t0_run
    # print(f'Grid search elapsed time: {t_run} seconds')
    # print(results)
    #
    # t0_save = t.time()
    # results.to_hdf('/nobackup/spanien1/salomon/EIC_spike_detection/GS_EIC_spike_rate_tschad.h5', key='Data')
    # t_save = t.time()-t0_save
    # print(f'Saving results elapsed time: {t_save} seconds')
    #
    # #############
    # # Load Data #
    # #############
    # # result_file = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/Computation_1/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5"
    # # results = read_cgs_results(result_file, key='Data')
    # # params = {'k_e': np.linspace(10., 30., 21), 'k_i': np.linspace(10., 30., 21)}
    # # T = 10.0
    # # dt = params['k_e'][1] - params['k_e'][0]
    #
    # ########
    # # Plot #
    # ########
    # plot_avrg_peak_dist(results, parameters=params, simulation_time=T, tick_size=5)
