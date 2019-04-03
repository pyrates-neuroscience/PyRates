import scipy.signal as sp
import matplotlib.pyplot as plt
from seaborn import cubehelix_palette

# PyRates imports
from pyrates.utility import plot_connectivity, plot_timeseries
from pyrates.cluster_compute.cluster_compute import *


def plot_example_ratios(fp, Cs, ei_ratio, io_ratio, ei_samples, io_samples, dts, n_cols, n_rows, overview=False):

    results = pd.read_hdf(fp, key='/Results/r_E0_df')

    if overview:
        # Parameters
        ############
        ei_samples_true = [ei_ratio[np.argmin(np.abs(ei_ratio - ei_s))] for ei_s in ei_samples]
        io_samples_true = [io_ratio[np.argmin(np.abs(io_ratio - io_s))] for io_s in io_samples]
        ei_io_pair_list = list(zip(ei_samples_true, io_samples_true))

        k_ee = np.zeros(len(ei_ratio)*len(io_ratio))
        k_ei = np.zeros_like(k_ee)
        k_ie = np.zeros_like(k_ee)
        k_ii = np.zeros_like(k_ee)

        n = 0
        k_ee += Cs
        for ei in ei_ratio:
            for io in io_ratio:
                k_ei[n] += Cs / (ei * io)
                k_ie[n] += Cs / io
                k_ii[n] += Cs / ei
                n += 1

        params = {'k_ee': k_ee, 'k_ei': k_ei, 'k_ie': k_ie, 'k_ii': k_ii}

        # Storer
        ########
        peaks_freq = np.zeros((len(ei_ratio), len(io_ratio)))
        peaks_amp = np.zeros_like(peaks_freq)
        time_series = []
        time_series_markers = []

        # Compute data
        ##############
        for k_ee_, k_ei_, k_ie_, k_ii_ in zip(params['k_ee'], params['k_ei'], params['k_ie'], params['k_ii']):
            if not results[k_ee_][k_ei_][k_ie_][k_ii_].isnull().any().any():
                data = results[k_ee_][k_ei_][k_ie_][k_ii_].loc[100.0:]
                r = np.argmin(np.abs(ei_ratio - k_ee_ / k_ii_))
                c = np.argmin(np.abs(io_ratio - k_ee_ / k_ie_))
                if (ei_ratio[r], io_ratio[c]) in ei_io_pair_list:
                    time_series.append(data)
                    time_series_markers.append([r, c])
                data = np.array(data)
                peaks, props = sp.find_peaks(data.squeeze(), prominence=0.6 * (np.max(data) - np.mean(data)))
                if len(peaks) > 1:
                    diff = np.mean(np.diff(peaks)) * dts * 0.01
                    peaks_freq[r, c] = 1 / diff
                    peaks_amp[r, c] = np.mean(props['prominences'])

        mask = peaks_amp > 0.1
        peaks_freq_masked = peaks_freq * mask

        # Plot
        ######
        fig, ax = plt.subplots(ncols=n_cols+1, nrows=n_rows, figsize=(20, 15), gridspec_kw={})
        step_tick_x, step_tick_y = int(peaks_freq.shape[1]/10), int(peaks_freq.shape[0]/10)

        # Plot peak rate
        cm1 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=2.5, rot=-0.1)
        cax1 = plot_connectivity(peaks_freq, ax=ax[0, 0], yticklabels=list(np.round(ei_ratio, decimals=2)),
                                 xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm1)
        for n, label in enumerate(ax[0, 0].xaxis.get_ticklabels()):
            if n % step_tick_x != 0:
                label.set_visible(False)
        for n, label in enumerate(ax[0, 0].yaxis.get_ticklabels()):
            if n % step_tick_y != 0:
                label.set_visible(False)
        cax1.set_xlabel('intra/inter pcs')
        cax1.set_ylabel('exc/inh pcs')
        cax1.set_title(f'max freq (C = {Cs})')
        for [ei_c, io_c] in time_series_markers:
            cax1.plot(io_c, ei_c, 'x', color='red', markersize='10')

        # Plot magnitude
        cm2 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=-2.0, rot=-0.1)
        cax2 = plot_connectivity(peaks_amp, ax=ax[1, 0], yticklabels=list(np.round(ei_ratio, decimals=2)),
                                 xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm2)
        for n, label in enumerate(ax[1, 0].xaxis.get_ticklabels()):
            if n % step_tick_x != 0:
                label.set_visible(False)
        for n, label in enumerate(ax[1, 0].yaxis.get_ticklabels()):
            if n % step_tick_y != 0:
                label.set_visible(False)
        cax2.set_xlabel('intra/inter pcs')
        cax2.set_ylabel('exc/inh pcs')
        cax2.set_title(f'mean peak amp (C = {Cs})')
        for [ei_c, io_c] in time_series_markers:
            cax2.plot(io_c, ei_c, 'x', color='white', markersize='10')

        # Plot mask
        cm3 = cubehelix_palette(n_colors=int(len(ei_ratio) * len(io_ratio)), as_cmap=True, start=3.5, rot=-0.1)
        cax3 = plot_connectivity(peaks_freq_masked, ax=ax[2, 0], yticklabels=list(np.round(ei_ratio, decimals=2)),
                                 xticklabels=list(np.round(io_ratio, decimals=2)), cmap=cm3)
        for n, label in enumerate(ax[2, 0].xaxis.get_ticklabels()):
            if n % step_tick_x != 0:
                label.set_visible(False)
        for n, label in enumerate(ax[2, 0].yaxis.get_ticklabels()):
            if n % step_tick_y != 0:
                label.set_visible(False)
        cax3.set_xlabel('intra/inter pcs')
        cax3.set_ylabel('exc/inh pcs')
        cax3.set_title(f'max freq masked (C = {Cs})')
        for [ei_c, io_c] in time_series_markers:
            cax3.plot(io_c, ei_c, 'x', color='black', markersize='10')

        # Plot time series
        ax_col = list(range(1, n_cols+1))*n_rows
        cax = list(range(len(ax_col)))
        for n in range(len(time_series)):
            row = int(np.floor(n/n_cols))
            col = ax_col[n]
            cax[n] = plot_timeseries(time_series[n], ax=ax[row, col])
            ei_val = ei_ratio[time_series_markers[n][0]]
            io_val = io_ratio[time_series_markers[n][1]]
            cax[n].set_title(f'{n}: ei={np.round(ei_val, decimals=2)}; io={np.round(io_val, decimals=2)}')

    else:
        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(20, 15), gridspec_kw={})
        ax_col = list(range(n_cols)) * n_rows

        cax = list(range(len(ax_col)))
        for n, (ei_c, io_c) in enumerate(zip(ei_samples, io_samples)):
            ei = ei_ratio[np.argmin(np.abs(ei_ratio - ei_c))]
            io = io_ratio[np.argmin(np.abs(io_ratio - io_c))]
            k_ee = Cs
            k_ei = Cs / (ei * io)
            k_ie = Cs / io
            k_ii = Cs / ei
            data = results[k_ee][k_ei][k_ie][k_ii].loc[100:]
            row = int(np.floor(n / n_cols))
            col = ax_col[n]
            cax[n] = plot_timeseries(data, ax=ax[row, col])
            cax[n].set_title(f'{n}: ei={np.round(ei, decimals=2)}; io={np.round(io, decimals=2)}')

    plt.suptitle(f'EI-circuit sensitivity to population Coupling strengths (pcs), alpha=0, C={Cs}')
    plt.show()


if __name__ == "__main__":

    ei_ratio = np.linspace(0.5, 4.0, 101)
    io_ratio = np.linspace(0.5, 4.0, 101)
    dts = 1e-2

    # file = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/Coupling_alpha_0_high_res/Results/" \
    #        "DefaultGrid_0/CGS_result_DefaultGrid_0.h5"
    # ei_examples = [1.5, 0.9, 2.5, 3.5, 2.5, 2.5]
    # io_examples = [1.2, 1.5, 1.5, 1.5, 1.7, 1.2]
    # Cs = 15.0
    # plot_example_ratios(file, Cs=Cs, ei_ratio=ei_ratio, io_ratio=io_ratio, ei_samples=ei_examples,
    #                     io_samples=io_examples, dts=dts, n_cols=3, n_rows=2, overview=True)

    file = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/Coupling_alpha_0_high_res/Results/" \
           "DefaultGrid_3/CGS_result_DefaultGrid_3.h5"
    ei_examples = [1.2, 1.2, 1.2, 2.4, 2.4, 2.4, 3.6, 3.6, 3.6]
    io_examples = [1.2, 2.2, 3.2, 1.2, 2.2, 3.2, 1.2, 2.2, 3.2]
    Cs = 30.0
    plot_example_ratios(file, Cs=Cs, ei_ratio=ei_ratio, io_ratio=io_ratio, ei_samples=ei_examples,
                        io_samples=io_examples, dts=dts, n_cols=3, n_rows=3, overview=True)

    # file = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/EIC/Coupling_alpha_0_high_res/Results/" \
    #        "DefaultGrid_1/CGS_result_DefaultGrid_1.h5"
    # ei_examples = [1.0, 1.0, 1.0, 2.4, 2.4, 2.4, 3.6, 3.6, 3.6]
    # io_examples = [1.2, 1.7, 2.2, 1.2, 1.7, 2.2, 1.2, 1.7, 2.2]
    # Cs = 20.0
    # plot_example_ratios(file, Cs=Cs, ei_ratio=ei_ratio, io_ratio=io_ratio, ei_samples=ei_examples,
    #                     io_samples=io_examples, dts=dts, n_cols=3, n_rows=3, overview=True)
