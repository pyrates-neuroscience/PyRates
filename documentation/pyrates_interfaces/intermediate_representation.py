"""
The Intermediate Representation - A Graph-Based Representation of the Model Definition
======================================================================================

In this tutorial, you will learn how to translate a PyRates model definition, which has either been implemented via the
YAML or the Python interface, into a network graph representation. This representation is called the *intermediate
representsation* in PyRates, since it provides the link between the model definition provided by the user and the
backend representation of the model equations that is used to perform simulations etc.
The *intermediate representation* allows the user to dynamically build up a network graph of nodes and edges from the
respective template classes. This network graph is based on the powerful graph tool *networkx*. More specifically,
each PyRates model is represented as a :code:`networkx.MultiDiGraph`. This representation comes with the full list of
advantages of a :node:`networkx` model, such as various algorithms for measurements of graph characteristics
(centralities, cluster coefficients, shortest paths, ...). For a detailed documentation of the :code:`networkx` package,
check out their `GitHub page <https://networkx.github.io/>`_.

Below, you will learn:

    - how to transform any PyRates template into an *intermediate representation*,
    - what the basic properties of the *intemediate representation* are,
    - how you can manipulate an *intermediate representation*.

In accordance with our model definition examples, we will base this tutorial on the Jansen-Rit model [1]_, which
describes the dynamic interactions between 3 neural populations via their average firing rates.
It has been used to describe the macroscopic, electrophysiological activity within a cortical column. A graphic
representation of such a model, placed inside a brain network, can be found in the figure below.

.. _fig1:

.. figure:: ../../../pyrates_interfaces/images/pyrates_model.png
   :width: 700

   Figure 1

The structure of the full Jansen-Rit model is depicted in :ref:`fig1` B. As can be seen, the interactions
between pyramidal cells (PCs), excitatory interneurons (EINs) and inhibitory interneurons (IINs) depend on 4 edges
between those populations: PCs to EINs, PCs to IINs, EINs to PCs and IINs to PCs. You will recognize this structure,
when we will have a look at the *intermediate representation* of our model.

References
^^^^^^^^^^

.. [1] B.H. Jansen & V.G. Rit (1995) *Electroencephalogram and visual evoked potential generation in a mathematical
       model of coupled cortical columns.* Biological Cybernetics, 73(4): 357-366.
"""

# %%
# Part 1: Translating a model definition into an intermediate representation
# --------------------------------------------------------------------------
#
# First, we will need to load a model template into PyRates. We choose to load a YAML-based model definition, but this
# can be done just as well from a Python-based model definition.

from pyrates.frontend import CircuitTemplate
jr_template = CircuitTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.JRC")

print(jr_template.nodes)
print(jr_template.edges)

# %%
# As demonstrated by the :code:`print()` calls above, the :code:`CircuitTemplate` contains fields for nodes and edges,
# which contain list of :code:`NodeTemplate` and :code:`EdgeTemplate` instances with :code:`OperatorTemplate` instances
# defined on them that contain the underlying model equations. This template can be transformed into an intermediate
# representation via the :code:`apply()` method defined for each template class:

jr_ir = jr_template.apply()

print(jr_ir.nodes)
print(jr_ir.edges)

# %%
# Notice that the :code:`nodes` and :code:`edges` attributes still exist, but contain instances of :code:`networkx`
# objects now. For a detailed documentation of the *intermediate representation* structure, check out the API for the
# :code:`CircuitIR` class.  During the transformation into a :code:`CircuitIR` instance, the user-specified sources
# and targets of all edges have been realized and the operator hierarchies on all nodes and edges have been inferred
# automatically as well. You can check this at single nodes:

print(jr_ir.nodes['PC']['node'].op_graph.nodes)
print(jr_ir.nodes['PC']['node'].op_graph.edges)

# %%
# The above example demonstrates that operator hierarchies at nodes and edges are represented by :code:`networkx`
# graphs as well. In this specific example, the PCs are defined by 4 operators. Two synaptic rate-to-potential operators
# (RPO_e_pc and RPO_i), that provide the input for a potential-to-rate operator (PRO) and observer operator (OBS).
# To learn about the mathematical details behind these operators, check out the tutorial for YAML-based or Python-based
# model definitions, or the model introduction of the Jansen-Rit model.
# Similarly, you can check whether edges have been translated to the graph properly via:

print(jr_ir.edges[('EIN', 'PC', 0)])

# %%
# The output of this :code:`print()` call reveals that there exists an edge from EINs to PCs on :code:`jr_ir` with a
# scalar weight of :code:`weight=108.0` that maps from the variable :code:`m_out` of the operator :code:`PRO` on the
# EIN population to the variable :code:`m_in` of the operator :code:`RPO_e_pc` on the PC population.
#
# As an additional check whether the network graph looks as expected, you can simply draw the graph:

from pyrates.utility.visualization import plot_network_graph
import matplotlib.pyplot as plt

plot_network_graph(jr_ir)
plt.show()

# %%
# Part 2: Modifying an Existing Graph Representation
# --------------------------------------------------
#
# After you created a :code:`CircuitIR` instance, you can dynamically add nodes, edges and even full circuits. This can
# be done via the following methods, defined for the :code:`CircuitIR` class:
#
#   - :code:`.add_nodes_from(node_list)` --> adds a collection of nodes to the graph
#   - :code:`.add_node(node)` --> add a single node to the graph
#   - :code:`.add_edges_from(edge_list)` --> adds a collection of edges to the graph
#   - :code:`.add_edge(edge)` --> add a single edge to the graph
#   - :code:`add_circuit(circuit)` --> add a circuit (consisting of nodes and edges) to the existing one
#   - :code:`add_edges_from_matrix()` --> method that allows to add edges to the network via a connectivity matrix.
#
# As an example, lets create a network of coupled Jansen-Rit models from a random connectivity matrix:

import numpy as np
from pyrates.ir import CircuitIR

n_jrcs = 10  # number of Jansen-Rit models in network
k = 20.0  # connection strength scaling
connectivity = np.random.uniform(0.0, 1.0, (10, 10)) * k   # connectivity matrix
jr_network = CircuitIR(label='jansen-rit network')   # base CircuitIR instance

# add all jansen-rit models to the network
node_labels = [f'jrc_{i}' for i in range(n_jrcs)]
for i in range(n_jrcs):
    jr_network.add_circuit(label=node_labels[i],
                           circuit=CircuitTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.JRC"))

# connect jansen-rit networks via connectivity matrix
jr_network.add_edges_from_matrix(source_var='PC/PRO/m_out', target_var='PC/RPO_e_pc/m_in', nodes=node_labels,
                                 weight=connectivity)

print(jr_network.nodes)
print(jr_network.edges)

# %%
# Part 3: Analyzing the network graph
# -----------------------------------
#

