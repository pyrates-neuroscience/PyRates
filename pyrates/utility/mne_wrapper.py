
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network model_templates and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
# 
# Copyright (C) 2017-2018 the original authors (Richard Gast and 
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain 
# Sciences ("MPI CBS") and contributors
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
# 
# CITATION:
# 
# Richard Gast and Daniel Rose et. al. in preparation
"""Provides functions to build MNE objects (raw, epoch, evoked) from PyRates simulation results (csv) or from circuit
objects.
"""

# external packages
import numpy as np
from typing import Union, Optional, List, Any
from pandas import DataFrame

# pyrates internal imports

# meta infos
__author__ = 'Richard Gast'
__status__ = 'Development'


############################################################
# wrapper to build an mne object from observer information #
############################################################


def mne_from_dataframe(sim_results: DataFrame,
                       ch_types: Union[str, List[str]] = 'eeg',
                       ch_names: Optional[Union[str, List[str]]] = None,
                       events: Optional[np.ndarray] = None,
                       event_keys: Optional[List[str]] = None,
                       epoch_start: Optional[float] = None,
                       epoch_end: Optional[float] = None,
                       epoch_duration: Optional[float] = None,
                       epoch_averaging: bool = False,
                       ) -> Any:
    """Uses the data stored on a circuit to create a raw/epoch/evoked mne object.

    Parameters
    ----------
    sim_results
        Pandas dataframe in which circuit simulation results are stored (output of circuit's `run` function).
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
    Any
        MNE object that contains either the raw, epoched, or averaged (over epochs) data.

    """

    import mne

    # extract information from arguments
    ####################################

    dt = sim_results.index[1] - sim_results.index[0]

    # channel information
    if not ch_names:
        ch_names = list(sim_results.keys())
    if type(ch_types) is str:
        ch_types = [ch_types for _ in range(len(ch_names))]
    ch_names_str = []
    for name in ch_names:
        ch_names_str.append(str(name))

    # epoch/event information
    if not epoch_start:
        epoch_start = -0.2 if not epoch_duration else 0.
    if not epoch_end:
        epoch_end = 0.5 if not epoch_duration else epoch_duration - 1/dt

    # create raw mne object
    #######################

    # create mne info object
    info = mne.create_info(ch_names=ch_names_str, ch_types=ch_types, sfreq=1/dt)

    # create raw object from info and circuit data
    raw = mne.io.RawArray(data=sim_results[ch_names].values.T, info=info)

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
                 ) -> Any:
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
    Any
        MNE object that contains either the raw, epoched, or averaged (over epochs) data.

    """

    import mne

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
