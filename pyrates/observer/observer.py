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
            self.target_populations = [pop for pop in circuit.populations.keys()]
        else:
            self.target_populations = target_populations
        if not target_states:
            self.target_states = ['membrane_potential', 'firing_rate']
        else:
            self.target_states = target_states

        self.states = [[[] for _ in self.target_populations] for __ in self.target_states]

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
            target_states = list(self.states[target_populations[0]].keys())
        self.precision = int(np.log10(1 / self.sampling_step_size)) + 2

        # state dictionary
        for state in target_states:
            if state not in self.states.keys():
                self.states[state] = dict()
                for pop in target_populations:
                    self.states[state][pop] = [None] * len(self.time)
            else:
                for pop in target_populations:
                    if pop not in self.states[state].keys():
                        self.states[state][pop] = [None] * len(self.time)

    def store_state_variables(self, circuit: object):
        """Goes through all target populations and adds the target state variables to the observer.
        """

        if circuit.t % self.sampling_step_size < self.precision:

            for i, state in enumerate(self.target_states):

                for j, pop in enumerate(self.target_populations):
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
                 target_state: str = 'membrane_potential'
                 ):
        """Instantiates external observer.
        """

        # set attributes
        ################

        self.sampling_step_size = observer.sampling_step_size
        self.states = DataFrame.from_dict(observer.states)
        self.times = observer.time
        self.population_labels = observer.states[list(observer.states.keys())[0]].keys()

        # reduce states to indicated populations
        ########################################

        # check target populations and population weights
        if not target_populations:
            target_populations = [['PCs']]
            target_population_weights = [[1.0]]
        if not target_population_weights:
            target_population_weights = [[1.0 for __ in range(len(target_populations[0]))]
                                         for _ in range(len(target_populations))]

        # loop over all groups of target populations
        states_new = list()
        for i, target_group in enumerate(target_populations):

            # loop over each population in group
            states_group = list()
            for j, target in enumerate(target_group):

                # get all populations that contain target label and weight them as indicated
                idx = [k for k, test in enumerate(self.population_labels) if target in test]
                states_group.append(self.states[:, idx] * target_population_weights[i][j])

            # calculate weighted average over grouped populations
            states_new.append(np.sum(np.array(states_group), axis=0))

        # create new states array
        states_new = np.array(states_new)
        self.states = np.reshape(states_new, (states_new.shape[1], states_new.shape[0]*states_new.shape[2]))

    def observe(self,
                store_observations: bool=False,
                filename: Optional[str] = None,
                path: Optional[str] = None,
                time_window: Optional[list] = None
                ) -> np.ndarray:
        """Generates observation data from population states.
        """

        if time_window:
            start = int(time_window[0] / self.sampling_step_size)
            stop = int(time_window[1] / self.sampling_step_size)
        else:
            start = 0
            stop = int(self.times[-1] / self.sampling_step_size)

        output_length = stop - start

        output = np.zeros((output_length, self.states.shape[1]))
        for t in range(start, stop):
            output[t, :] = self.states[t, :]

        return output
