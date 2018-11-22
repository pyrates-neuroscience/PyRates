from pyrates.frontend import CircuitIR, CircuitTemplate
from pyrates.backend import ComputeGraph
from matplotlib.pyplot import *

# parameters
n_jrcs = 2

# circuit IR setup
circuit_temp = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.circuit.JansenRitCircuit")
circuits = {}
for n in range(n_jrcs):
    circuits['jrc.' + str(n)] = circuit_temp
circuit_ir = CircuitIR.from_circuits('jrc_net', circuits=circuits).network_def(revert_node_names=True)

# create backend
net = ComputeGraph(circuit_ir, vectorize='ops')
#inp_pc = 220. + np.random.randn(3000, n_jrcs) * 0.
#inp_in = np.zeros((3000, n_jrcs))
results, _ = net.run(simulation_time=3.,
                     outputs={'v': ('all', 'JansenRitPRO.0', 'V')},
 #                    inputs={('JR_PC', 'JansenRitExcitatorySynapseRCO.0', 'u'): inp_pc,
 #                            ('JR_IIN', 'JansenRitExcitatorySynapseRCO.0', 'u'): inp_in,
 #                            ('JR_EIN', 'JansenRitExcitatorySynapseRCO.0', 'u'): inp_in})
                     )
results.pop('time')
results.plot()
show()
