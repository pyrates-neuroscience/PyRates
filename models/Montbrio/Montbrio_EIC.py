from pyrates.utility import grid_search, plot_timeseries, plot_psd, plot_connectivity
import time as t
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


def plot_peaks_per_second(data, parameters, T, tick_size=5):
    num_peaks = np.zeros([len(parameters['k_e']), len(parameters['k_i'])])
    for m, k_e in enumerate(parameters['k_e']):
        for n, k_i in enumerate(parameters['k_i']):
            data = np.array(results[k_e][k_i])
            peaks = sp.argrelextrema(data, np.greater)
            num_peaks[m, n] = int(len(peaks[0]) / T)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    plot_connectivity(num_peaks, ax=ax, xticklabels=list(parameters['k_i']), yticklabels=list(parameters['k_e']))

    # Show only every 25th tick (steps of 5.0 on the axis label)
    step_size_x = parameters['k_e'][1] - parameters['k_e'][0]
    step_size_y = parameters['k_i'][1] - parameters['k_i'][0]
    step_tick_x = tick_size/step_size_x
    step_tick_y = tick_size/step_size_y
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        label.font_size = 20
        if n % 2 != 0:
            label.set_visible(False)

    plt.tick_params(labelsize=20)
    axis_font = {'fontname': 'Arial', 'size': '25'}
    ax.set_xlabel('k_i', fontdict=axis_font)
    ax.set_ylabel('k_e', fontdict=axis_font)
    ax.set_title('Average spikes per second', fontdict=axis_font)

    fig.savefig("/data/hu_salomon/Documents/MA/Graphics/Plots/Montbrio_EIC_spike_rate_3", format="svg")
    plt.show()


circuit_template = "Montbrio.EI_Circuit"

T = 2.0

param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
                     'nodes': ['E.0', 'I.0']},
             'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
                     'nodes': ['E.0', 'I.0']}
             }

# params = {'k_e': np.linspace(10., 30., 3), 'k_i': np.linspace(10., 30., 3)}
# params = {'k_e': np.linspace(10., 15., 11), 'k_i': np.linspace(25., 30., 11)}
params = {'k_e': np.linspace(15., 20., 21), 'k_i': np.linspace(15., 20., 21)}


###############
# Grid Search #
###############
t0_run = t.time()
results = grid_search(circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                      param_grid=params,
                      param_map=param_map,
                      dt=1e-5,
                      simulation_time=T,
                      inputs={},
                      outputs={"r": ("E", "Op_e.0", "r")},
                      sampling_step_size=1e-3,
                      permute_grid=False)

t_run = t.time()-t0_run
print(f'Grid search elapsed time: {t_run} seconds')
print(results)
# t0_save = t.time()
# results.to_hdf('/nobackup/spanien1/salomon/EIC_spike_detection/GS_EIC_spike_rate_tschad.h5', key='Data')
# t_save = t.time()-t0_save
# print(f'Saving results elapsed time: {t_save} seconds')

###############
# Grid Search #
###############
# plot_peaks_per_second(results, parameters=params, T=T, tick_size=1)
