"""Provides functions to build MNE objects (raw, epoch, evoked) from PyRates simulation results (csv) or from circuit
objects.
"""

# external packages
import mne
import numpy as np
from typing import Union, Optional, List

# pyrates internal imports
from pyrates.observer import CircuitObserver, EEGMEGObserver

# meta infos
__author__ = 'Richard Gast'
__status__ = 'Development'


############################################################
# wrapper to build an mne object from observer information #
############################################################


def mne_from_observer(observer: Union[CircuitObserver, EEGMEGObserver],
                      ch_types: Union[str, List[str]] = 'eeg',
                      ch_names: Optional[Union[str, List[str]]] = None,
                      target_variable: str = 'membrane_potential',
                      events: Optional[np.ndarray] = None,
                      event_keys: Optional[List[str]] = None,
                      epoch_start: Optional[float] = None,
                      epoch_end: Optional[float] = None,
                      epoch_duration: Optional[float] = None,
                      epoch_averaging: bool = False,
                      ) -> Union[mne.io.Raw, mne.Epochs, mne.Evoked]:
    """Uses the data stored on a circuit to create a raw/epoch/evoked mne object.

    Parameters
    ----------
    observer
        Instance of an :class:`CircuitObserver` (found on every :class:`Circuit` instance on which `run()` was called)
        or an :class:`EEGMEGObserver` (needs to be instantiated from an internal observer).
    ch_types
        Type of the channels, the observation time-series of the observers refer to.
    ch_names
        Name of each channel/observation time-series.
    target_variable
        State variable that is to be extracted from the observer. Only needed, if the observer is a
        :class:'CircuitObserver'.
    events
        2D array defining events during the simulation. For a more detailed documentation, see the docstring for
        parameter `events` of :class:`mne.Epochs`.
    event_keys
        Names of the events. For a more detailed documentation, see the docstring for parameter `event_id` of
        :class:`mne.Epochs`.
    epoch_start
        Time, relative to event onset, at which an epoch should start [unit = s]. For a more detailed documentation,
        see the docstring for parameter `tmin` of :class:`mne.Epochs`.
    epoch_end
        Time, relative to event onset, at which an epoch should end [unit = s]. For a more detailed documentation,
        see the docstring for parameter `tmax` of :class:`mne.Epochs`.
    epoch_duration
        Instead of passing `events`, this parameter can be used to create epochs with a fixed duration [unit = s].
        If this is used, do not pass `epoch_start` or `epoch_end`. For a more detailed documentation,
        see the docstring for parameter `duration` of :function:`mne.make_fixed_length_events`.
    epoch_averaging
        Only relevant, if `events` or `event_duration` were passede. If true, an :class:`mne.EvokedArray` instance will
        be returned that contains ttime-series averaged over all epochs.

    Returns
    -------
    Union[mne.io.Raw, mne.Epochs, mne.Evoked]
        MNE object that contains either the raw, epoched, or averaged (over epochs) data.

    """

    # extract information from arguments
    ####################################

    # circuit information
    if type(observer) is CircuitObserver:
        states = np.array(observer.states)
        states = states[observer.target_states.index(target_variable), :, :].squeeze()
    else:
        states = observer.observe(store_observations=False).values.T

    sfreq = 1/observer.sampling_step_size

    # channel information
    if not ch_names:
        ch_names = observer.population_labels
    if type(ch_types) is str:
        ch_types = [ch_types for _ in range(len(ch_names))]

    # epoch/event information
    if not epoch_start:
        epoch_start = -0.2 if not epoch_duration else 0.
    if not epoch_end:
        epoch_end = 0.5 if not epoch_duration else epoch_duration - 1/sfreq

    # create raw mne object
    #######################

    # create mne info object
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # create raw object from info and circuit data
    raw = mne.io.RawArray(data=states, info=info)

    # create final mne object
    #########################

    # if events or event information is passed, create Epoch or Evoked object
    if events is not None or epoch_duration:

        # check whether events still have to be created or not
        if events is None:
            events = mne.make_fixed_length_events(raw=raw, id=0, duration=epoch_duration)

        # check whether event labels still have to be created or not
        if not event_keys:
            event_keys = dict()
            for event in np.unique(events[:, 2]):
                event_keys['event_' + str(event)] = event

        # create epoch object from raw data and event information
        mne_object = mne.Epochs(raw=raw, events=events, event_id=event_keys, tmin=epoch_start, tmax=epoch_end)

        # create Evoked object by averaging over epochs if epoch_averaging is true
        if epoch_averaging:

            # average over epoch data
            data = mne_object.get_data()
            n_epochs = len(data)
            data = np.mean(data, axis=0)

            # create evoked object
            mne_object = mne.EvokedArray(data=data, info=info, tmin=epoch_start, comment=event_keys['event_0'],
                                         nave=n_epochs)

    # stick with Raw object
    else:

        mne_object = raw

    return mne_object


###################################################
# wrapper to build an mne object from output file #
###################################################


def mne_from_csv(csv_dir: str,
                 ch_types: Union[str, List[str]] = 'eeg',
                 ch_names: Optional[Union[str, List[str]]] = None,
                 events: Optional[np.ndarray] = None,
                 event_keys: Optional[List[str]] = None,
                 epoch_start: Optional[float] = None,
                 epoch_end: Optional[float] = None,
                 epoch_duration: Optional[float] = None,
                 epoch_averaging: bool = False,
                 ) -> Union[mne.io.Raw, mne.Epochs, mne.Evoked]:
    """Uses the data stored on circuit to create a raw/epoch/evoked mne object.

    Parameters
    ----------
    csv_dir
        Full path + filename of the csv file that contains the circuit outputs from which an MNE object should be
        created.
    ch_types
        Type of the channels, the observation time-series of the observers refer to.
    ch_names
        Name of each channel/observation time-series.
    events
        2D array defining events during the simulation. For a more detailed documentation, see the docstring for
        parameter `events` of :class:`mne.Epochs`.
    event_keys
        Names of the events. For a more detailed documentation, see the docstring for parameter `event_id` of
        :class:`mne.Epochs`.
    epoch_start
        Time, relative to event onset, at which an epoch should start [unit = s]. For a more detailed documentation,
        see the docstring for parameter `tmin` of :class:`mne.Epochs`.
    epoch_end
        Time, relative to event onset, at which an epoch should end [unit = s]. For a more detailed documentation,
        see the docstring for parameter `tmax` of :class:`mne.Epochs`.
    epoch_duration
        Instead of passing `events`, this parameter can be used to create epochs with a fixed duration [unit = s].
        If this is used, do not pass `epoch_start` or `epoch_end`. For a more detailed documentation,
        see the docstring for parameter `duration` of :function:`mne.make_fixed_length_events`.
    epoch_averaging
        Only relevant, if `events` or `event_duration` were passede. If true, an :class:`mne.EvokedArray` instance will
        be returned that contains ttime-series averaged over all epochs.

    Returns
    -------
    Union[mne.io.Raw, mne.Epochs, mne.Evoked]
        MNE object that contains either the raw, epoched, or averaged (over epochs) data.

    """

    # extract information from arguments
    ####################################

    # from data file
    from pandas.io.parsers import read_csv

    output = read_csv(csv_dir, delim_whitespace=True, header=0)
    states = output.values
    sfreq = states[-1, 0] / states.shape[0]
    states = states[:, 1:]
    n_channels = states.shape[1]

    # channel information
    if type(ch_types) is str:
        ch_types = [ch_types for _ in range(n_channels)]
    if not ch_names:
        ch_names = output.keys()
        ch_names.pop(0)

    # epoch/event information
    if not epoch_start:
        epoch_start = -0.2 if not epoch_duration else 0.
    if not epoch_end:
        epoch_end = 0.5 if not epoch_duration else epoch_duration - 1 / sfreq

    # create raw mne object
    #######################

    # create mne info object
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # create raw object from info and circuit data
    raw = mne.io.RawArray(data=states.T, info=info)

    # create final mne object
    #########################

    # if events or event information is passed, create Epoch or Evoked object
    if events is not None or epoch_duration:

        # check whether events still have to be created or not
        if events is None:
            events = mne.make_fixed_length_events(raw=raw, id=0, duration=epoch_duration)

        # check whether event labels still have to be created or not
        if not event_keys:
            event_keys = dict()
            for event in np.unique(events[:, 2]):
                event_keys['event_' + str(event)] = event

        # create epoch object from raw data and event information
        mne_object = mne.Epochs(raw=raw, events=events, event_id=event_keys, tmin=epoch_start, tmax=epoch_end)

        # create Evoked object by averaging over epochs if epoch_averaging is true
        if epoch_averaging:

            # average over epoch data
            data = mne_object.get_data()
            n_epochs = len(data)
            data = np.mean(data, axis=0)

            # create evoked object
            mne_object = mne.EvokedArray(data=data, info=info, tmin=epoch_start, comment=event_keys['event_0'],
                                         nave=n_epochs)

    # stick with Raw object
    else:

        mne_object = raw

    return mne_object
