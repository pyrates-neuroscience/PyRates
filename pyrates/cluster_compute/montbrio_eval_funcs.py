# System imports

# External imports
from pyrates.utility import plot_connectivity
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


def plot_avrg_peaks_per_second(results, parameters, simulation_time, tick_size=5, fp=None):
    num_peaks = np.zeros([len(parameters['k_e']), len(parameters['k_i'])])
    for m, k_e in enumerate(parameters['k_e']):
        for n, k_i in enumerate(parameters['k_i']):
            data = np.array(results[k_e][k_i])
            peaks = sp.argrelextrema(data, np.greater)
            num_peaks[m, n] = int(len(peaks[0]) / simulation_time)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    plot_connectivity(num_peaks, ax=ax, xticklabels=list(parameters['k_i']), yticklabels=list(parameters['k_e']))

    # Show only every tick_size tick on the axis
    step_size_x = parameters['k_e'][1] - parameters['k_e'][0]
    step_size_y = parameters['k_i'][1] - parameters['k_i'][0]
    step_tick_x = np.round(tick_size/step_size_x)
    step_tick_y = np.round(tick_size/step_size_y)
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % step_tick_x != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        label.font_size = 20
        if n % step_tick_y != 0:
            label.set_visible(False)

    plt.tick_params(labelsize=20)
    axis_font = {'fontname': 'Arial', 'size': '25'}
    ax.set_xlabel('k_i', fontdict=axis_font)
    ax.set_ylabel('k_e', fontdict=axis_font)
    ax.set_title('Average peaks per second', fontdict=axis_font)

    if fp:
        fig.savefig(f'{fp}/average_peaks_per_sec', format="svg")

    plt.show()
    return num_peaks


def plot_avrg_peak_dist(results, parameters, dt, tick_size=5, fp=None):
    peak_dist = np.zeros([len(parameters['k_e']), len(parameters['k_i'])])
    for m, k_e in enumerate(parameters['k_e']):
        for n, k_i in enumerate(parameters['k_i']):
            data = np.array(results[k_e][k_i])
            peaks = sp.argrelextrema(data, np.greater)
            diff = np.diff(peaks[0])
            diff = np.mean(diff)*dt
            # peak_dist[m, n] = simulation_time / (diff*dt)
            peak_dist[m, n] = 1 / diff

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
    plot_connectivity(peak_dist, ax=ax, xticklabels=list(parameters['k_i']), yticklabels=list(parameters['k_e']))

    # Show only every tick_size tick on the axis
    step_size_x = parameters['k_e'][1] - parameters['k_e'][0]
    step_size_y = parameters['k_i'][1] - parameters['k_i'][0]
    step_tick_x = np.round(tick_size/step_size_x)
    step_tick_y = np.round(tick_size/step_size_y)
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % step_tick_x != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        label.font_size = 20
        if n % step_tick_y != 0:
            label.set_visible(False)

    plt.tick_params(labelsize=20)
    axis_font = {'fontname': 'Arial', 'size': '25'}
    ax.set_xlabel('k_i', fontdict=axis_font)
    ax.set_ylabel('k_e', fontdict=axis_font)

    ax.set_title('Average time between peaks', fontdict=axis_font)

    if fp:
        fig.savefig(f'{fp}/average_peak_dist', format="svg")

    plt.show()
    return peak_dist


if __name__ == "__main__":
    pass
