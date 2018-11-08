from pyrates.frontend import CircuitIR, CircuitTemplate
from pyrates.backend import ComputeGraph

# parameters
n_jrcs = 10

# circuit IR setup
circuit_temp = CircuitTemplate.from_yaml("pyrates.frontend.circuit.templates.JansenRitCircuit")
circuits = {}
for n in range(n_jrcs):
    circuits['jrc' + str(n)] = circuit_temp
circuit_ir = CircuitIR.from_circuits('jrc_net', circuits=circuits).network_def()

# create backend
net = ComputeGraph(circuit_ir)
results = net.run(1.)
