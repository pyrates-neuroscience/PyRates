import sys
import numpy as np
sys.path.append('../base')
import nmm_network as nmm
from matplotlib.pyplot import *

##########################
# set network parameters #
##########################

positions = np.zeros((3, 3))
connections = np.ones((3, 3, 2)) * 70
connections[:, 2, 0] = 0
connections[:, 0:2, 1] = 0
populations = ['JansenRitExcitatoryPopulation', 'JansenRitExcitatoryPopulation', 'JansenRitInhibitoryPopulation']
synapses = ['AMPA_current', 'GABAA_current']
axons = ['Knoesche', 'Knoesche', 'Knoesche']
velocities = np.array([2, 2, 2])
step_size = 0.001
kernel_length = 1000
stimulation_time = 3

######################
# initialize network #
######################

NMM = nmm.NeuralMassModel(connections, population_labels=populations, synapses=synapses, axons=axons,
                          positions=positions, velocities=velocities, step_size=step_size,
                          synaptic_kernel_length=kernel_length)

############################################
# simulate population behavior under input #
############################################

synaptic_input = np.random.rand(np.int(stimulation_time / step_size), 3, 2) * 200

NMM.run(synaptic_input, stimulation_time, verbose=True)
states = NMM.neural_mass_states

###########################
# plot simulation results #
###########################

figure()
plot(np.squeeze(states[0, :]))
show()
