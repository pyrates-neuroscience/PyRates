"""
Benchmark: legacy scalar-edge API vs. new PopulationTemplate + Connectivity API.

Network: N Wilson-Cowan excitatory nodes, all-to-all coupled with a random
weight matrix W (diagonal = 15.0, off-diagonal ~ N(0, 1/sqrt(N))).

Run with:
    conda run -n sbi python examples/benchmark_population_connectivity.py
"""

import sys
import os
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ensure the development version of PyRates (with PopulationTemplate) is used
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# reset any cached imports so the dev version takes precedence
for mod in list(sys.modules.keys()):
    if mod.startswith("pyrates"):
        del sys.modules[mod]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rng(seed=42):
    return np.random.default_rng(seed)


def _make_weights(N: int, rng, off_diag_scale: float = 1.0) -> np.ndarray:
    """NxN coupling matrix: diagonal = 15.0, off-diagonal ~ N(0, scale/sqrt(N))."""
    W = rng.normal(0.0, off_diag_scale / np.sqrt(N), (N, N))
    np.fill_diagonal(W, 15.0)
    return W


# ---------------------------------------------------------------------------
# legacy approach
# ---------------------------------------------------------------------------

def build_legacy(N: int, W: np.ndarray) -> "CircuitTemplate":
    """Build a legacy circuit: N individual nodes + N² scalar edges."""
    from pyrates.frontend.template import from_yaml, CircuitTemplate
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    exc_pop = from_yaml("model_templates.neural_mass_models.wilsoncowan.exc_pop")
    nodes = {f"e{k}": exc_pop for k in range(N)}
    circuit = CircuitTemplate(name=f"wc_legacy_n{N}", nodes=nodes)

    node_names = [f"e{k}" for k in range(N)]
    circuit.add_edges_from_matrix(
        source_var="rate_op/r",
        target_var="se_op/r_in",
        source_nodes=node_names,
        weight=W,
        min_weight=0.0,   # include all entries so the connectivity is identical
    )
    return circuit


def run_legacy(circuit, T: float, dt: float) -> float:
    """Run legacy circuit and return wall time of .run() in seconds."""
    t0 = time.perf_counter()
    circuit.run(
        simulation_time=T,
        step_size=dt,
        solver="euler",
        outputs={"r": "e0/rate_op/r"},
        clear=True,
        verbose=False,
    )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# new approach
# ---------------------------------------------------------------------------

def build_new(N: int, W: np.ndarray) -> "CircuitTemplate":
    """Build a new-API circuit: one PopulationTemplate + one Connectivity."""
    from pyrates.frontend.template import from_yaml, CircuitTemplate
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    exc_pop = from_yaml("model_templates.neural_mass_models.wilsoncowan.exc_pop")
    pop = PopulationTemplate(name="e", node=exc_pop, n=N)
    conn = Connectivity(
        source="e/rate_op/r",
        target="e/se_op/r_in",
        weights=W,
    )
    circuit = CircuitTemplate(
        name=f"wc_new_n{N}",
        populations={"e": pop},
        connections=[conn],
    )
    return circuit


def run_new(circuit, T: float, dt: float) -> float:
    """Run new-API circuit and return wall time of .run() in seconds."""
    t0 = time.perf_counter()
    circuit.run(
        simulation_time=T,
        step_size=dt,
        solver="euler",
        outputs={"r": "e/rate_op/r"},
        clear=True,
        verbose=False,
    )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# benchmark runner
# ---------------------------------------------------------------------------

def benchmark(N_values: list, T: float = 5.0, dt: float = 1e-3, seed: int = 42):
    rng = _rng(seed)

    rows = []
    header = f"{'N':>6}  {'legacy build (s)':>17}  {'legacy run (s)':>15}  " \
             f"{'new build (s)':>14}  {'new run (s)':>12}  {'speedup':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for N in N_values:
        W = _make_weights(N, rng)

        # --- legacy ---
        t0 = time.perf_counter()
        leg_circuit = build_legacy(N, W)
        t_leg_build = time.perf_counter() - t0

        t_leg_run = run_legacy(leg_circuit, T, dt)

        # --- new ---
        t0 = time.perf_counter()
        new_circuit = build_new(N, W)
        t_new_build = time.perf_counter() - t0

        t_new_run = run_new(new_circuit, T, dt)

        speedup = (t_leg_build + t_leg_run) / (t_new_build + t_new_run)
        row = dict(N=N, t_leg_build=t_leg_build, t_leg_run=t_leg_run,
                   t_new_build=t_new_build, t_new_run=t_new_run, speedup=speedup)
        rows.append(row)

        print(f"{N:>6}  {t_leg_build:>17.3f}  {t_leg_run:>15.3f}  "
              f"{t_new_build:>14.3f}  {t_new_run:>12.3f}  {speedup:>8.1f}x")

    print(sep)
    return rows


# ---------------------------------------------------------------------------
# optional plot
# ---------------------------------------------------------------------------

def plot_results(rows: list):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    Ns = [r["N"] for r in rows]
    leg_total = [r["t_leg_build"] + r["t_leg_run"] for r in rows]
    new_total = [r["t_new_build"] + r["t_new_run"] for r in rows]
    leg_build = [r["t_leg_build"] for r in rows]
    new_build = [r["t_new_build"] for r in rows]
    leg_sim   = [r["t_leg_run"]   for r in rows]
    new_sim   = [r["t_new_run"]   for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(Ns, leg_total, "o-", label="Legacy (total)", color="steelblue")
    ax.plot(Ns, new_total, "s-", label="New (total)",    color="tomato")
    ax.plot(Ns, leg_build, "o--", label="Legacy (build)", color="steelblue", alpha=0.5)
    ax.plot(Ns, new_build, "s--", label="New (build)",    color="tomato",    alpha=0.5)
    ax.plot(Ns, leg_sim,   "o:", label="Legacy (sim)",   color="steelblue", alpha=0.3)
    ax.plot(Ns, new_sim,   "s:", label="New (sim)",      color="tomato",    alpha=0.3)
    ax.set_xlabel("N (population size)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time vs. N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    speedups = [r["speedup"] for r in rows]
    ax.plot(Ns, speedups, "D-", color="seagreen")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("N (population size)")
    ax.set_ylabel("Speedup (legacy / new)")
    ax.set_title("Speedup of new API over legacy")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Wilson-Cowan network: legacy scalar-edge vs. PopulationTemplate+Connectivity",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("benchmark_wc_population.png", dpi=150)
    print("Plot saved to benchmark_wc_population.png")
    plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sweep N from small to 100 (the user's target).
    # N=100 legacy: ~10 000 scalar edges — compilaton may take a few minutes.
    N_values = [10, 50, 100, 500]

    print("\nWilson-Cowan all-to-all coupling benchmark")
    print("  T = 5.0 s,  dt = 1 ms,  fully dense N×N weight matrix\n")

    rows = benchmark(N_values, T=5.0, dt=1e-3)
    plot_results(rows)
