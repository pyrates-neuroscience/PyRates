"""Provides functions to build MNE objects (raw, epoch, evoked) from PyRates simulation results (csv) or from circuit
objects.
"""

__author__ = 'Richard Gast'
__status__ = 'Development'


import mne
import numpy as np
from typing import Union, Optional, List
from pyrates.observer import CircuitObserver, EEGMEGObserver


############################################################
# wrapper to build an mne object from observer information #
############################################################


def mne_from_observer(observer: Union[CircuitObserver, EEGMEGObserver],
                      ch_types: Union[str, List[str]] = 'eeg',
                      ch_names: Optional[Union[str, List[str]]] = None,
                      events: Optional[np.ndarray] = None,
                      event_labels: Optional[List[str]] = None,
                      epoch_start: Optional[float] = None,
                      epoch_end: Optional[float] = None,
                      epoch_duration: Optional[float] = None,
                      epoch_averaging: bool = False,
                      ) -> Union[mne.io.Raw, mne.Epochs, mne.Evoked]:
    """Uses the data stored on circuit to create a raw/epoch/evoked mne object.
    """

    # extract information from arguments
    ####################################

    # circuit information
    if type(observer) is CircuitObserver:
        states = np.array(observer.states['membrane_potential']).T
    else:
        states = observer.observe(store_observations=False).T

    sfreq = 1/observer.sampling_step_size
    n_channels = states.shape[0]

    # channel information
    if type(ch_types) is str:
        ch_types = [ch_types for _ in range(n_channels)]
    if not ch_names:
        ch_names = observer.population_labels

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
        if not event_labels:
            event_labels = dict()
            for event in np.unique(events[:, 2]):
                event_labels['event_' + str(event)] = event

        # create epoch object from raw data and event information
        mne_object = mne.Epochs(raw=raw, events=events, event_id=event_labels, tmin=epoch_start, tmax=epoch_end)

        # create Evoked object by averaging over epochs if epoch_averaging is true
        if epoch_averaging:

            # average over epoch data
            data = mne_object.get_data()
            n_epochs = len(data)
            data = np.mean(data, axis=0)

            # create evoked object
            mne_object = mne.EvokedArray(data=data, info=info, tmin=epoch_start, comment=event_labels['event_0'],
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
                 event_labels: Optional[List[str]] = None,
                 epoch_start: Optional[float] = None,
                 epoch_end: Optional[float] = None,
                 epoch_duration: Optional[float] = None,
                 epoch_averaging: bool = False,
                 ) -> Union[mne.io.Raw, mne.Epochs, mne.Evoked]:
    """Uses the data stored on circuit to create a raw/epoch/evoked mne object.
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
        if not event_labels:
            event_labels = dict()
            for event in np.unique(events[:, 2]):
                event_labels['event_' + str(event)] = event

        # create epoch object from raw data and event information
        mne_object = mne.Epochs(raw=raw, events=events, event_id=event_labels, tmin=epoch_start, tmax=epoch_end)

        # create Evoked object by averaging over epochs if epoch_averaging is true
        if epoch_averaging:

            # average over epoch data
            data = mne_object.get_data()
            n_epochs = len(data)
            data = np.mean(data, axis=0)

            # create evoked object
            mne_object = mne.EvokedArray(data=data, info=info, tmin=epoch_start, comment=event_labels['event_0'],
                                         nave=n_epochs)

    # stick with Raw object
    else:

        mne_object = raw

    return mne_object
