r"""
Large-Scale Networks with PopulationTemplate and Connectivity
=============================================================

Building large networks of coupled neural populations is a core workflow in computational
neuroscience. With the *legacy* PyRates API, a network of :math:`N` all-to-all coupled
nodes requires defining :math:`N^2` individual scalar edges, which becomes impractically
slow for :math:`N \gtrsim 50` because each edge spawns its own operator subgraph during
compilation.

**PyRates 1.1.0** introduces two new classes that replace the :math:`O(N^2)` edge loop with
a single matrix object:

- :code:`PopulationTemplate` — treats :math:`N` identical dynamical units as one vectorised
  entity. Variables are stored as length-:math:`N` vectors from the start, so downstream
  code never needs to loop over individual nodes.
- :code:`Connectivity` — expresses coupling between two population variables as a single
  :math:`(N_{\text{target}} \times N_{\text{source}})` weight matrix, optionally decorated
  with a transmission delay or delay distribution.

Throughout this tutorial we build an all-to-all coupled network of Wilson-Cowan excitatory
populations and demonstrate:

1. Basic population definition and homogeneous matrix coupling.
2. Heterogeneous per-unit parameters.
3. Delayed coupling via a scalar :code:`delays` argument.
4. Build-time scaling: a brief comparison against the legacy edge API.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from pyrates import CircuitTemplate, NodeTemplate, clear
from pyrates.frontend.template.population import PopulationTemplate, Connectivity
from pyrates.ir.node import clear_ir_caches

# %%
# We will use the Wilson-Cowan excitatory population as the single unit.
# It contains two operators:
#
# - :code:`rate_op`: integrates the firing rate :math:`r' = (-r + (1 - kr) m) / \tau`
# - :code:`se_op`: the sigmoid input transform :math:`m = \sigma(s(r_{in} + r_{ext} - \theta))`
#
# In the all-to-all network, the output :code:`rate_op/r` of every unit feeds into the
# input :code:`se_op/r_in` of every other unit (including itself) via the weight matrix.

exc_pop = NodeTemplate.from_yaml("model_templates.neural_mass_models.wilsoncowan.exc_pop")

# simulation parameters shared by all sections
T = 2.0
dt = 1e-3
rng = np.random.default_rng(42)

# %%
#
# Section 1: Basic usage
# ----------------------
#
# Define a population of N = 10 Wilson-Cowan units and connect them with a random
# weight matrix whose diagonal (self-coupling) is set to 15.0 so the units activate.

N = 10
W = rng.normal(0.0, 1.0 / np.sqrt(N), (N, N))
np.fill_diagonal(W, 15.0)

# --- define population and connectivity ---
pop = PopulationTemplate(name="e", node=exc_pop, n=N)
conn = Connectivity(source="e/rate_op/r", target="e/se_op/r_in", weights=W)

# --- assemble circuit ---
circuit = CircuitTemplate(name="wc_basic", populations={"e": pop}, connections=[conn])

# --- simulate ---
results = circuit.run(
    simulation_time=T, step_size=dt, solver="euler",
    outputs={"r": "e/rate_op/r"}, clear=True,
)

# --- plot: individual trajectories + population mean ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(results, alpha=0.6, linewidth=0.8)
axes[0].set_title(f"Individual firing rates (N = {N})")
axes[0].set_xlabel("time")
axes[0].set_ylabel("r")

axes[1].plot(results.mean(axis=1), color="steelblue", linewidth=1.5)
axes[1].fill_between(results.index, results.min(axis=1), results.max(axis=1),
                     alpha=0.25, color="steelblue", label="min/max range")
axes[1].set_title("Population mean firing rate")
axes[1].set_xlabel("time")
axes[1].set_ylabel("mean r")
axes[1].legend()
fig.tight_layout()
plt.show()

clear_ir_caches()

# %%
# The equivalent legacy definition would require creating N individual node entries and
# calling :code:`add_edges_from_matrix`, which registers :math:`N^2` scalar edges:
#
# .. code-block:: python
#
#     nodes = {f"e{k}": exc_pop for k in range(N)}
#     circuit_legacy = CircuitTemplate(name="wc_legacy", nodes=nodes)
#     circuit_legacy.add_edges_from_matrix(
#         source_var="rate_op/r",
#         target_var="se_op/r_in",
#         source_nodes=list(nodes.keys()),
#         weight=W,
#         min_weight=0.0,
#     )
#
# For N = 10 that is already 100 edges; for N = 100 it is 10 000 edges and compilation
# can take several minutes. The :code:`PopulationTemplate` + :code:`Connectivity` path
# stays fast regardless of N because the weight matrix is passed directly to the vectorised
# backend — see Section 4 for a timing comparison.

# %%
#
# Section 2: Heterogeneous per-unit parameters
# ---------------------------------------------
#
# Neural populations are rarely perfectly homogeneous. The :code:`params` argument of
# :code:`PopulationTemplate` accepts a dictionary mapping :code:`'op/var'` keys to either
# a scalar (broadcast to all units) or an array of length :math:`N` (one value per unit).
#
# Here we draw individual time constants :math:`\tau_i` from a uniform distribution around
# the default value of 10 ms, giving each unit a slightly different integration speed.

tau_values = rng.uniform(8.0, 12.0, size=N)
pop_het = PopulationTemplate(
    name="e",
    node=exc_pop,
    n=N,
    params={"rate_op/tau": tau_values},
)
conn_het = Connectivity(source="e/rate_op/r", target="e/se_op/r_in", weights=W)
circuit_het = CircuitTemplate(name="wc_het", populations={"e": pop_het}, connections=[conn_het])

results_het = circuit_het.run(
    simulation_time=T, step_size=dt, solver="euler",
    outputs={"r": "e/rate_op/r"}, clear=True,
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results.mean(axis=1), label=r"homogeneous ($\tau = 10$)")
ax.plot(results_het.mean(axis=1), label=r"heterogeneous ($\tau \sim \mathcal{U}(8, 12)$)",
        linestyle="dashed")
ax.set_xlabel("time")
ax.set_ylabel("population mean firing rate")
ax.set_title("Effect of heterogeneous time constants")
ax.legend()
plt.show()

clear_ir_caches()

# %%
#
# Section 3: Delayed coupling
# ----------------------------
#
# Axonal conduction delays can be added by passing a scalar :code:`delays` argument to
# :code:`Connectivity`. This introduces the same ring-buffer mechanism as an edge with a
# :code:`delay` attribute in the legacy API, but applied uniformly to the entire weight
# matrix in a single operation.

conn_delayed = Connectivity(
    source="e/rate_op/r",
    target="e/se_op/r_in",
    weights=W,
    delays=0.5,   # uniform 0.5-unit propagation delay for all connections
)
circuit_delay = CircuitTemplate(name="wc_delay", populations={"e": pop}, connections=[conn_delayed])

results_delay = circuit_delay.run(
    simulation_time=T, step_size=dt, solver="euler",
    outputs={"r": "e/rate_op/r"}, clear=True,
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results.mean(axis=1), label="no delay")
ax.plot(results_delay.mean(axis=1), label="delay = 0.5", linestyle="dashed")
ax.set_xlabel("time")
ax.set_ylabel("population mean firing rate")
ax.set_title("Effect of uniform propagation delay")
ax.legend()
plt.show()

clear_ir_caches()

# %%
# Distributed delays (gamma-kernel convolution) are also supported via the :code:`spread`
# argument, following the same convention as the legacy edge API:
#
# .. code-block:: python
#
#     conn_gamma = Connectivity(
#         source="e/rate_op/r",
#         target="e/se_op/r_in",
#         weights=W,
#         delays=0.5,   # mean delay
#         spread=0.1,   # variance of the gamma kernel
#     )

# %%
#
# Section 4: Build-time scaling
# ------------------------------
#
# The key advantage of the new API is that circuit build time stays nearly constant as
# :math:`N` grows, while the legacy approach scales as :math:`O(N^2)`.

N_values = [5, 10, 25, 50]
t_legacy = []
t_new = []

for N_bench in N_values:
    clear_ir_caches()
    W_bench = np.random.default_rng(0).normal(0.0, 1.0 / np.sqrt(N_bench), (N_bench, N_bench))
    np.fill_diagonal(W_bench, 15.0)

    # legacy: N² scalar edges
    t0 = time.perf_counter()
    nodes_b = {f"e{k}": exc_pop for k in range(N_bench)}
    c_leg = CircuitTemplate(name=f"wc_leg_{N_bench}", nodes=nodes_b)
    c_leg.add_edges_from_matrix(
        source_var="rate_op/r",
        target_var="se_op/r_in",
        source_nodes=list(nodes_b.keys()),
        weight=W_bench,
        min_weight=0.0,
    )
    t_legacy.append(time.perf_counter() - t0)
    clear_ir_caches()

    # new: single Connectivity object
    t0 = time.perf_counter()
    pop_b = PopulationTemplate("e", exc_pop, N_bench)
    conn_b = Connectivity("e/rate_op/r", "e/se_op/r_in", weights=W_bench)
    c_new = CircuitTemplate(name=f"wc_new_{N_bench}", populations={"e": pop_b}, connections=[conn_b])
    t_new.append(time.perf_counter() - t0)
    clear_ir_caches()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(N_values, t_legacy, "o-", label="legacy (N² edges)", color="steelblue")
axes[0].plot(N_values, t_new, "s-", label="PopulationTemplate + Connectivity", color="tomato")
axes[0].set_xlabel("N (population size)")
axes[0].set_ylabel("build time (s)")
axes[0].set_title("Circuit build time vs. population size")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

speedups = [tl / tn for tl, tn in zip(t_legacy, t_new)]
axes[1].plot(N_values, speedups, "D-", color="seagreen")
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("N (population size)")
axes[1].set_ylabel("speedup (legacy / new)")
axes[1].set_title("Build-time speedup")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Legacy scalar-edge API vs. PopulationTemplate + Connectivity", fontsize=11)
fig.tight_layout()
plt.show()
