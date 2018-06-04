"""Runs JRC with tensorflow on population basis
"""

# imports
#########

import tensorflow as tf
from pyrates.population import Population, SynapticInputPopulation
from matplotlib.pyplot import *
import time as t

# parameter definition
######################

# general
step_size = 1e-3
simulation_time = 1.0
n_steps = int(simulation_time / step_size)
c = 135.
d = 0.0 * step_size
inp_mean = 220.
inp_var = 22.
inp = inp_mean + np.random.randn(n_steps) * inp_var

# tensorflow graph setup
########################

# initialize graph
gr = tf.Graph()

# initialize populations
PCs = Population(synapses=['JansenRitExcitatorySynapse', 'JansenRitInhibitorySynapse'], axon='JansenRitAxon',
                 step_size=step_size, tf_graph=gr, key='pcs', max_population_delay=d)
EINs = Population(synapses=['JansenRitExcitatorySynapse'], axon='JansenRitAxon', step_size=step_size, tf_graph=gr,
                  key='eins', max_population_delay=d)
IINs = Population(synapses=['JansenRitExcitatorySynapse'], axon='JansenRitAxon', step_size=step_size, tf_graph=gr,
                  key='iins', max_population_delay=d)
INP = SynapticInputPopulation(inp, tf_graph=gr)

# add connections
PCs.connect(EINs.synapses['eins_JansenRitExcitatorySynapse'], 1.0 * c, int(d/step_size))
PCs.connect(IINs.synapses['iins_JansenRitExcitatorySynapse'], 0.25 * c, int(d/step_size))
EINs.connect(PCs.synapses['pcs_JansenRitExcitatorySynapse'], 0.8 * c, int(d/step_size))
IINs.connect(PCs.synapses['pcs_JansenRitInhibitorySynapse'], 0.25 * c, int(d/step_size))
INP.connect(PCs.synapses['pcs_JansenRitExcitatorySynapse'], 1.0, 0)

# add grouping + projection operations to graph
with gr.as_default():

    with tf.variable_scope('jrc1'):

        pass_inf = tf.group(PCs.connect_to_targets(),
                            EINs.connect_to_targets(),
                            IINs.connect_to_targets(),
                            name='pass_infos')
        with tf.control_dependencies([pass_inf]):
            pass_inp = INP.connect_to_targets()
        pass_all = tf.group(pass_inf, pass_inp, name='pass_all')

        state_update = tf.tuple([PCs.state_update, EINs.state_update, IINs.state_update, INP.state_update],
                                name='state_updates')

    init = tf.global_variables_initializer()

# run session
#############

states = []

# initalize session
with tf.Session(graph=gr) as sess:

    # initialize output storage
    writer = tf.summary.FileWriter('/tmp/log/', graph=gr)

    # initialize variables
    sess.run(init)

    t_start = t.time()

    # perform simulation
    for _ in range(n_steps-1):

        # perform step with input
        sess.run(pass_all)
        sess.run(state_update)

        # save states
        states.append([PCs.membrane_potential.eval(sess),
                       EINs.membrane_potential.eval(sess),
                       IINs.membrane_potential.eval(sess),
                       ]
                      )

    t_end = t.time()

    writer.close()

# summary
print(str(simulation_time) + ' s of JRC behavior was simulated in ' + str(t_end - t_start) + ' s with a step size of ' +
      str(step_size) + ' s.')
fig, ax = subplots()
ax.plot(np.array(states))
legend(['PCs', 'EINs', 'IINs'])
ax.set_title('JRC states')
ax.set_ylabel('v')
ax.set_xlabel('step')
fig.show()
