"""This file includes the base observer class for the population & circuit level.
"""

import numpy as np
from typing import Union, Optional, List

__status__ = "development"
__author__ = "Richard Gast"


#########################
# base circuit observer #
#########################


class CircuitObserver(object):
    """Base circuit observer class. Manages how the population states are observed.
    """

    def __init__(self,
                 circuit: object,
                 sampling_step_size: Optional[float] = None,
                 target_populations: Optional[Union[np.ndarray, list]] = None,
                 target_states: Optional[List[str]] = None
                 ) -> None:
        """Instantiates base circuit observer.
        """

        self.sampling_step_size = sampling_step_size if sampling_step_size else circuit.step_size
        self.target_populations = target_populations if target_populations else range(circuit.n_populations)
        self.target_states = target_states if target_states else ['membrane_potential']
        self.states = dict()
        for target in self.target_states:
            self.states[target] = list()
        self.times = list()
        self.precision = int(np.log10(1 / self.sampling_step_size)) + 2
        self.population_labels = [pop.label for pop in circuit.populations]

    def update(self,
               circuit: object,
               sampling_step_size: Optional[float] = None,
               target_populations: Optional[Union[np.ndarray, list]] = None,
               target_states: Optional[List[str]] = None
               ) -> None:
        """Updates observer system to prepare a new run.
        """

        self.sampling_step_size = sampling_step_size if sampling_step_size else circuit.step_size
        self.target_populations = target_populations if target_populations else range(circuit.n_populations)
        self.precision = int(np.log10(1 / self.sampling_step_size)) + 2

        # state dictionary
        if target_states:
            self.target_states = target_states
            for target in target_states:
                if target not in self.states.keys():
                    self.states[target] = [0.] * len(self.states['membrane_potential'])

    def store_state_variables(self, circuit: object):
        """Goes through all target populations and adds the target state variables to the observer.
        """

        if circuit.t % self.sampling_step_size < self.precision:

            for key in self.target_states:

                states_tmp = list()
                for p in self.target_populations:
                    states_tmp.append(circuit.populations[p].state_variables[key])

                self.states[key].append(states_tmp)

            self.times.append(circuit.t)

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
        self.states = np.array(observer.states[target_state])
        self.times = observer.times
        self.population_labels = observer.population_labels

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
