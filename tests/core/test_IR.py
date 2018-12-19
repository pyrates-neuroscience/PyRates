"""
"""

__author__ = "Daniel Rose"
__status__ = "Development"


def test_move_edge_ops_to_nodes():
    """Test, if apply() functions all work properly"""

    path = "pyrates.examples.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.circuit import CircuitTemplate
    from pyrates.ir.circuit import CircuitIR

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()  # type: CircuitIR
    circuit2 = circuit.move_edge_operators_to_nodes()

    for source, target, data in circuit2.edges(data=True):
        # check that no operators are left in the edges of the rearranged circuit
        assert len(data["edge_ir"].op_graph) == 0

        # check that operator from previous edges is indeed in target nodes
        # original_edge = circuit.edges[(source, target, 0)]["edge_ir"]
        # original_op = list(original_edge.op_graph.nodes)[0]
        # assert f"{original_op}.0" in circuit2[target]
