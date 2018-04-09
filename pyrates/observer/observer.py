"""This file includes the base observer class for the population & circuit level.
"""

import numpy as np
from typing import Union, Optional, List
from pandas import DataFrame, concat, Series

__status__ = "development"
__author__ = "Richard Gast"


#########################
# base circuit observer #
#########################


class CircuitObserver(object):
    """Base circuit observer class. Manages how the population states are observed.

    Parameters
    ----------
    circuit
        Instance of :class:`pyrates.circuit.Circuit`, the observer will observe.
    sampling_step_size
        Step-size with which the circuit states will be stored (default = simulation step-size) [unit = s].
    target_populations
        Indices referring to the circuit populations, whose states should be observed. By default, all populations are.
    target_states
        Strings indicating the population state variables that should be observed. By default, only the membrane
        potential is.

    """

    def __init__(self,
                 circuit: object,
                 sampling_step_size: Optional[float] = None,
                 target_populations: Optional[List[str]] = None,
                 target_states: Optional[List[str]] = None
                 ) -> None:
        """Instantiates base circuit observer.
        """

        self.sampling_step_size = sampling_step_size if sampling_step_size else circuit.step_size
        if not target_populations:
            self.population_labels = [pop for pop in circuit.populations.keys() if 'dummy' not in pop]
        else:
            self.population_labels = target_populations
        if not target_states:
            self.target_states = ['membrane_potential']
        else:
            self.target_states = target_states

        self.states = [[[] for _ in self.population_labels] for __ in self.target_states]

        self.precision = int(np.log10(1 / self.sampling_step_size)) + 2
        self.time = list()

    def update(self,
               circuit: object,
               sampling_step_size: Optional[float] = None,
               target_populations: Optional[List[str]] = None,
               target_states: Optional[List[str]] = None
               ) -> None:
        """Updates observer system to prepare a new run.
        """

        self.sampling_step_size = sampling_step_size if sampling_step_size else circuit.step_size
        if not target_populations:
            target_populations = [pop for pop in circuit.populations.keys()]
        if not target_states:
            target_states = self.target_states
        self.precision = int(np.log10(1 / self.sampling_step_size)) + 2

        # state dictionary
        for i, state in enumerate(target_states):
            if state not in self.target_states:
                state_list = list()
                for pop in target_populations:
                    state_list.append([None] * len(self.time))
                self.states.append(state_list)
            else:
                for pop in target_populations:
                    if pop not in self.population_labels:
                        self.states[i].append([None] * len(self.time))

    def store_state_variables(self, circuit: object):
        """Goes through all target populations and adds the target state variables to the observer.
        """

        if circuit.t % self.sampling_step_size < self.precision:

            for i, state in enumerate(self.target_states):

                for j, pop in enumerate(self.population_labels):
                    self.states[i][j].append(getattr(circuit.populations[pop], state))

            self.time.append(circuit.t)

    def clear(self):
        """Clears state history stored on observer.
        """

        for state in self.states:
            for pop in state:
                pop.clear()

##########################
# base external observer #
##########################


class ExternalObserver(object):
    """Base external observation class. Manages how the circuit dynamics can be observed from outside the system.
    """

    def __init__(self,
                 observer: CircuitObserver,
                 target_populations: Optional[list] = None,
                 target_population_weights: Optional[Union[List[list], list]] = None,
                 target_state: str = 'membrane_potential',
                 group_labels: Optional[list] = None
                 ):
        """Instantiates external observer.
        """

        # set attributes
        ################

        # general
        self.time = observer.time
        self.sampling_step_size = observer.sampling_step_size
        self.population_labels = observer.population_labels

        # create dataframe including all states plus the simulation time
        state_idx = observer.target_states.index(target_state)
        self.states = DataFrame(data=np.array(observer.states[state_idx]).T, columns=observer.population_labels)

        # reduce states to indicated populations
        ########################################

        # check target populations and population weights
        if not target_populations:
            target_populations = [[pop] for pop in self.population_labels]
            target_population_weights = [[1.0] for _ in self.population_labels]
        if not target_population_weights:
            target_population_weights = [[1.0 for __ in range(len(target_populations[0]))]
                                         for _ in range(len(target_populations))]

        # loop over all groups of target populations
        for i, target_group in enumerate(target_populations):

            target_col = list()

            # loop over each population in group
            for j, target in enumerate(target_group):

                # get all populations that contain target label and weight them as indicated
                target_pops = [key for k, key in enumerate(self.population_labels) if target in key]
                self.states.loc[:, target_pops].mul(target_population_weights[i][j])
                self.states[target + '_tmp'] = self.states.loc[:, target_pops].sum(axis=1)
                self.states.drop(columns=target_pops, inplace=True)

                target_col.append(target + '_tmp')

            # combine grouped populations into single column
            if group_labels:
                group_key = group_labels[i]
            else:
                group_key = target.split('_')[0]
                if group_key in self.states.keys():
                    group_key = group_key + '_' + str(i)

            self.states[group_key] = self.states.loc[:, target_col].sum(axis=1)
            self.states.drop(columns=target_col, inplace=True)

    def observe(self,
                store_observations: bool=False,
                filename: Optional[str] = None,
                path: Optional[str] = None,
                time_window: Optional[list] = None
                ) -> DataFrame:
        """Generates observation data from population states.
        """

        if time_window:
            start = int(time_window[0] / self.sampling_step_size)
            stop = int(time_window[1] / self.sampling_step_size)
        else:
            start = 0
            stop = int(self.time[-1] / self.sampling_step_size)

        return self.states.iloc[start:stop, :]
