"""Standalone benchmark script.

Run with ``python tests/benchmarks.py`` for the default backend, or pass
``--backends <b1>[,<b2>,...]`` to time the same workload against any subset of
PyRates' active backends.

Examples:
    python tests/benchmarks.py
    python tests/benchmarks.py --backends default,jax
    python tests/benchmarks.py --backends default,torch --n-runs 5
"""

import argparse


def benchmark_single_jr_circuit(n_runs: int = 10, backend: str = 'default') -> dict:
    """Time template loading and IR application for the Jansen-Rit circuit.

    Parameters
    ----------
    n_runs
        Number of repetitions; the median is reported.
    backend
        PyRates backend to apply the template against.  Forwarded as
        ``template.apply(backend=backend, ...)``.

    Returns
    -------
    dict
        Mapping from stage name to the list of observed durations (seconds).
    """

    import time
    from pyrates import CircuitTemplate
    from pyrates import clear
    import numpy as np

    path = "model_templates.neural_mass_models.jansenrit.JRC"

    timer_dict = dict(template_load=[], template_apply=[])

    for _ in range(n_runs):

        # time template loading (backend-agnostic)
        tic = time.perf_counter()
        template = CircuitTemplate.from_yaml(path)
        toc = time.perf_counter()
        timer_dict["template_load"].append(toc - tic)

        # time template application against the chosen backend
        tic = time.perf_counter()
        template.apply(verbose=False, backend=backend)
        toc = time.perf_counter()
        timer_dict["template_apply"].append(toc - tic)

        clear(template)

    print(f"Benchmark results [backend={backend}, n_runs={n_runs}]:")
    for key, times in timer_dict.items():
        median = np.median(times)
        print(f"  {key}: {median:.4f}s")

    return timer_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default="default",
        help="Comma-separated list of backends to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of repetitions per backend (default: %(default)s)",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        benchmark_single_jr_circuit(n_runs=args.n_runs, backend=b)


if __name__ == "__main__":
    main()
