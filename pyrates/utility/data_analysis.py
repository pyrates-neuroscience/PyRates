# external imports
import pandas as pd
import numpy as np

# pyrates internal imports

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


def functional_connectivity(data, metric='cov', **kwargs):
    """Calculate functional connectivity of node timeseries in data.

    Parameters
    ----------
    data
    metric
    kwargs

    Returns
    -------

    """

    if 'time' in data.columns.values:
        idx = data.pop('time')
        data.index = idx

    # calculate functional connectivity
    ###################################

    if metric == 'cov':

        # covariance
        fc = np.cov(data.values.T, **kwargs)

    elif metric == 'corr':

        # pearsson correlation coefficient
        fc = np.corrcoef(data.values.T, **kwargs)

    elif metric in 'cohcohyimcohplvppcplipli2_unbiasedwpliwpli2_debiased':

        # phase-based connectivtiy/synchronization measurement
        from mne.connectivity import spectral_connectivity
        fc, _, _, _, _ = spectral_connectivity(np.reshape(data.values.T, (1, data.shape[1], data.shape[0])),
                                               method=metric,
                                               sfreq=1./(data.index[1] - data.index[0]),
                                               **kwargs)
        fc = fc[:, :, 0]

    else:

        raise ValueError(f'FC metric is not supported by this function: {metric}. Check the documentation of the '
                         f'argument `metric` for valid options.')

    return fc


def analytic_signal(data, fmin, fmax, nodes=None, **kwargs):
    """

    Parameters
    ----------
    data
    fmin
    fmax
    kwargs

    Returns
    -------

    """

    if 'time' in data.columns.values:
        idx = data.pop('time')
        data.index = idx

    if nodes:
        if type(nodes[0]) is str:
            data = data.loc[:, nodes]
        else:
            data = data.iloc[:, nodes]

    # create mne raw data object
    from pyrates.utility import mne_from_dataframe
    raw = mne_from_dataframe(data)

    # bandpass filter the raw data
    filter_args = ['filter_length', 'l_trans_bandwidth', 'h_trans_bandwidth', 'n_jobs', 'method', 'iir_params',
                   'copy', 'phase', 'fir_window', 'fir_design', 'pad', 'verbose']
    kwargs_tmp = {}
    for key in kwargs.keys():
        if key in filter_args:
            kwargs_tmp[key] = kwargs.pop(key)
    raw.filter(l_freq=fmin, h_freq=fmax, **kwargs_tmp)

    # apply hilbert transform
    raw.apply_hilbert()

    # get phase of analytic signal
    def get_angle(x):
        return np.angle(x) + np.pi
    raw_phase = raw.copy()
    raw_phase.apply_function(get_angle)
    raw_phase.apply_function(np.real, dtype=np.float32)
    raw_phase.apply_function(np.unwrap)

    # get amplitude of analytic signal
    raw_amplitude = raw.copy()
    raw_amplitude.apply_function(np.abs)
    raw_amplitude.apply_function(np.real, dtype=np.float32)

    # combine phase and amplitude into dataframe
    time = data.index
    data_phase = raw_phase.to_data_frame(scalings={'eeg': 1.})
    data_phase['time'] = time
    data_amp = raw_amplitude.to_data_frame(scalings={'eeg': 1.})
    data_amp['time'] = time
    data = pd.melt(data_phase, id_vars=['time'], var_name='node', value_name='phase')
    data_tmp = pd.melt(data_amp, id_vars=['time'], var_name='node', value_name='amplitude')
    data['amplitude'] = data_tmp['amplitude']

    return data
