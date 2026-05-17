# Backend Consistency Review

*Audit of [`pyrates/backend/`](pyrates/backend/) after the tensorflow removal and
JAX addition.  Five active backends are compared along three axes: interface
parity, code style, and performance.  Findings are ordered from most-impactful
(correctness / silent-failure) to nice-to-have.*

Backends in scope:
- [`BaseBackend`](pyrates/backend/base/base_backend.py) — numpy (default)
- [`TorchBackend`](pyrates/backend/torch/torch_backend.py)
- [`JaxBackend`](pyrates/backend/jax/jax_backend.py) — newly added
- [`FortranBackend`](pyrates/backend/fortran/fortran_backend.py)
- [`JuliaBackend`](pyrates/backend/julia/julia_backend.py)
- [`MatlabBackend`](pyrates/backend/matlab/matlab_backend.py) — *inherits from* `JuliaBackend`

---

## 1. Critical findings

### 1.1  Backends silently fall back to **numpy** for several operations
The `*_funcs` registries are *merged on top* of `base_funcs`, so any key not
overridden in the subclass falls through with `numpy.*` imports.  The following
keys are missing in both `torch_funcs` and `jax_funcs`:

```
broadcast_post, broadcast_pre, index, index_axis, index_range,
interp_rows, no_op, past, wsum
```

Impact per backend:

| Operation     | Torch effect                                                | JAX effect                                                                     |
| ------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `wsum`        | `np.einsum` on a torch tensor → CPU conversion, breaks grad | `np.einsum` inside a jit-traced function → **runtime error / leaks out of JIT** |
| `interp_rows` | numpy `interp` + `array` → silent CPU fallback              | same — breaks JIT                                                              |
| `broadcast_*` | `x[None,:]` is dtype-agnostic — works                       | works                                                                          |
| `index_*`     | pure Python `x[i]` — works for both                         | works                                                                          |
| `past`        | no-op pass-through — works                                  | works                                                                          |

Most simple models never hit `wsum` / `interp_rows`, which is why this hasn't
shown up in the test suite.  **Population-connectivity models hit `wsum`** —
the new examples in [`examples/benchmark_population_connectivity.py`](examples/benchmark_population_connectivity.py)
would silently lose GPU acceleration if used with the torch backend.

**Recommendation (quick win):** add explicit `wsum` / `interp_rows` entries to
`torch_funcs.py` and `jax_funcs.py` using `torch.einsum` / `jnp.einsum` and
`torch.functional.interp` / `jnp.interp`.

### 1.2  Torch backend doesn't override the DDE path
`BaseBackend._solve` routes DDEs to `_solve_scipy_dde`, which uses
`scipy.integrate.ode` and writes to a numpy `state_rec` buffer
([`base_backend.py:558`](pyrates/backend/base/base_backend.py#L558)).  If a torch
user calls `solver='scipy'` on a model with `delay` edges, the buffer-mutation
path uses numpy primitives on torch tensors via `state_rec[i, :] = solver.y` —
this *works* by accident because `solver.y` is already numpy, but the returned
trajectory is plain ndarray, losing autograd.

**Recommendation:** either add an explicit `_solve_scipy_dde` override in
`TorchBackend` that keeps tensors throughout, or document the limitation.

### 1.3  `subprocess.run(... stdout=DEVNULL, stderr=DEVNULL)` silently hides f2py failures
[`fortran_backend.py:220`](pyrates/backend/fortran/fortran_backend.py#L220):

```python
subprocess.run(f"python -m numpy.f2py -c -m {self._fname} {file}", shell=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

Issues:
- `python` resolves to whatever `$PATH` provides — not guaranteed to be the env
  that hosts the numpy the rest of the process is using.
- `shell=True` with an interpolated path is a small command-injection surface
  if `self._fname` ever comes from user input.
- DEVNULL swallows compiler errors entirely; if f2py fails the subsequent
  `from {self._fname} import {self._fname}` raises an opaque `ImportError`.

**Recommendation:** use `[sys.executable, '-m', 'numpy.f2py', ...]`, drop
`shell=True`, capture stderr to a string, and re-raise on non-zero return.

### 1.4  `from numpy import f2py` at module load
[`fortran_backend.py:41`](pyrates/backend/fortran/fortran_backend.py#L41) imports
`f2py` at top level.  In numpy ≥ 2.0, f2py is now its own package and the
top-level import works only if `meson` / `setuptools` etc. are also available.
This adds noticeable startup cost (~50–100 ms) and a hard dependency just to
import the *fortran* submodule — which most users will never use.

**Recommendation:** move `from numpy import f2py` inside
`FortranBackend.__init__` or inside `generate_func` (lazy import).

### 1.5  `from julia.api import Julia` at import time
[`julia_backend.py:78`](pyrates/backend/julia/julia_backend.py#L78) — same
problem: importing `JuliaBackend` always tries to start a Julia runtime, even
just to look up the class.  This is currently masked because
`computegraph.py` only imports it inside the dispatch branch, but it means
`from pyrates.backend.julia import JuliaBackend` (e.g. in a test discovery
sweep) crashes if Julia isn't installed.

**Recommendation:** defer the `julia.api` import to `__init__`.

---

## 2. Interface consistency

### 2.1  `_solve` override surface differs widely

| Backend  | Overrides                              | Solver options actually plumbed through                        |
| -------- | -------------------------------------- | -------------------------------------------------------------- |
| Base     | `_solve_*`                             | `euler`, `heun`, `scipy`, `scipy` + DDE                        |
| Torch    | `_solve_euler`, `_solve_scipy`         | `euler` (broken — uses `idx` from outer scope, see 4.1), `scipy` |
| JAX      | full `_solve`                          | `euler`, `heun`, `scipy`, `diffrax`, scipy-DDE (delegated)     |
| Fortran  | full `_solve` (wraps func, delegates)  | inherits base                                                  |
| Julia    | full `_solve`                          | `julia_ode`, `julia_dde` (auto-detected), + delegates to base  |
| Matlab   | full `_solve` (wraps func, delegates)  | inherits base                                                  |

**Consequences:**
- `solver='heun'` is missing in `TorchBackend`: requests fall through to
  `BaseBackend._solve_heun` which runs on raw numpy and breaks autograd.
- There is no documented "what solvers does this backend support" listing.

**Recommendation:** add a class attribute `SUPPORTED_SOLVERS` on each backend
and validate at the top of `_solve`.  Cleaner UX than a `PyRatesException`
buried four frames deep.

### 2.2  `add_hist_arg` default flips across backends

| Backend  | Default                                                                                     |
| -------- | ------------------------------------------------------------------------------------------- |
| Base     | True (`add_hist_arg = kwargs.pop('add_hist_arg', True)`)                                    |
| Torch    | True (inherits)                                                                             |
| JAX      | True (inherits)                                                                             |
| Julia    | True (explicitly: `super().__init__(..., add_hist_arg=True, ...)`)                          |
| Matlab   | True (explicitly)                                                                           |
| Fortran  | inherits — but `generate_func_head` *also* takes `add_hist_func=False` default in subclass! |

[`fortran_backend.py:102`](pyrates/backend/fortran/fortran_backend.py#L102) hard-codes
`add_hist_func=False` whereas Matlab hard-codes `add_hist_func=True` at
[`matlab_backend.py:115`](pyrates/backend/matlab/matlab_backend.py#L115).
The semantics drift between the constructor flag and the per-call argument
is confusing.

**Recommendation:** keep one source of truth — derive `add_hist_func` from
`self.add_hist_arg` and the `is_dde` flag inside `generate_func_head`.

### 2.3  `_no_funcs` exclusion list lives only in BaseBackend
[`base_backend.py:181`](pyrates/backend/base/base_backend.py#L181) hard-codes
`['identity', 'index_1d', 'index_2d', 'index_range', 'index_axis']`.  It
controls whether `func_str` is appended to `_helper_funcs` — which matters for
Fortran (helper code goes into the module) and Julia/Matlab (no helper funcs
needed because the language has its own indexing).  Currently:

- Fortran inherits the list, which is correct — these are not regular funcs.
- Julia/Matlab inherit it too, but they handle helpers differently anyway, so
  it's mostly inert.

This is not broken, just confusing.  A subclass that defines new no-op funcs
has to override `_no_funcs` *and* keep the parent's entries.

**Recommendation:** turn `_no_funcs` into a class attribute and let subclasses
extend with `_no_funcs = BaseBackend._no_funcs + ['foo']`.

### 2.4  Decorator-application code is duplicated across `generate_func` overrides

Every subclass with its own `generate_func` (`JuliaBackend:149`, `MatlabBackend:136`,
`FortranBackend:186`) re-implements the trailing:

```python
decorator = kwargs.pop('decorator', None)
if decorator:
    decorator_kwargs = kwargs.pop('decorator_kwargs', dict())
    rhs_eval = decorator(rhs_eval, **decorator_kwargs)
return rhs_eval
```

**Recommendation:** factor into `BaseBackend._apply_decorator(rhs, **kwargs) -> Callable`
and call it from each subclass.

---

## 3. Performance

### 3.1  JAX `_solve_euler` / `_solve_heun` are not JIT-fused
Even though the RHS is `@jit`-compiled, the outer Python loop in
[`jax_backend.py:243`](pyrates/backend/jax/jax_backend.py#L243) calls it once
per step — each call is a separate XLA dispatch.  For long simulations this is
~100–1000× slower than a fused `lax.scan` over the time axis.

**Recommendation:** wrap the integration loop in `jax.lax.scan`, save the state
on every step (or every `store_step`), and return the stacked trajectory.
Requires giving up DDE-style history mutation, but `_solve_diffrax` already
provides the production path.

### 3.2  Torch `_solve_scipy` recreates tensors on every step
[`torch_backend.py:95-97`](pyrates/backend/torch/torch_backend.py#L95):

```python
def f(t, y):
    rhs = func(torch.tensor(t, dtype=dtype), torch.tensor(y, dtype=dtype), *args)
    return rhs.numpy()
```

`torch.tensor(...)` copies.  For numpy input, `torch.as_tensor(y)` returns a
zero-copy view and would shave a measurable fraction off the per-step cost on
larger networks.

### 3.3  `BaseBackend.generate_func` recompiles even when source is unchanged
[`base_backend.py:367-374`](pyrates/backend/base/base_backend.py#L367) always
runs `compile()` + `exec()` even if the on-disk file is byte-for-byte identical
to a previously compiled one.  For interactive use (sweep over parameters,
each call to `get_run_func` writes the same source for the same model) this
recompiles every time.

**Recommendation:** cache compiled modules keyed by the SHA-256 of `func_str`.

### 3.4  `DDEHistory.update` copies y on every step
[`base_backend.py:77-79`](pyrates/backend/base/base_backend.py#L77):

```python
def update(self, t: float, y: np.ndarray):
    self._t.append(float(t))
    self._y.append(y.copy())
```

The copy is correct (the solver may overwrite y in place after this call) but
for a 100k-step simulation this is 100k allocations of size-(n_state,).
Replacing the lists with pre-allocated ring buffers (or a `collections.deque`
with `maxlen`) would cut both allocations and lookup cost for the
`bisect_right` search in `__call__`.

### 3.5  `BaseBackend._solve_euler`/`_solve_heun` rebuild `state_rec` from scratch
Both methods build `state_rec` with `np.zeros((store_steps, ...))` and write
into it.  For float32 networks of 10k+ states the zero-fill dominates startup
time.  Could be created uninitialised (`np.empty`) since every row is written
before being read.

### 3.6  `JaxBackend.get_var` converts every constant to a JAX array
[`jax_backend.py:115-122`](pyrates/backend/jax/jax_backend.py#L115).  This is
correct, but it means every constant scalar is a `jnp.ndarray` rather than a
Python float.  Inside the jit-traced function, JAX has to treat each as a
runtime input and tracks it through the trace cache.  Wrapping the constants
with `jax.lax.stop_gradient` and / or marking them as static via
`functools.partial(jit, static_argnums=...)` would let the XLA compiler
constant-fold them.

---

## 4. Specific bugs & oddities

### 4.1  `TorchBackend._solve_euler` shadows `idx` and only works when `t0 == 0`
[`torch_backend.py:104-121`](pyrates/backend/torch/torch_backend.py#L104):

```python
def _solve_euler(func, args, T, dt, dts, y, idx):
    ...
    for step in torch.arange(int(idx), steps):
        if step % store_step == 0:
            state_rec[idx, :] = y
            idx += 1
        rhs = func(step, y, *args)
        ...
```

The parameter `idx` is documented as "the initial step index" but it's also
used as the *write cursor* into `state_rec`.  First iteration writes to
`state_rec[t0, :]`, which is wrong unless `t0 == 0`.  Compare with the base
version which uses two separate variables (`idx` for storage, `step` for
time).

### 4.2  `state_rec[idx, :] = y` with `idx += 1` inside the conditional
This pattern in both `BaseBackend._solve_euler` and `BaseBackend._solve_heun`
([`base_backend.py:502`](pyrates/backend/base/base_backend.py#L502)) has a subtle
boundary case: the first stored sample happens when `step % store_step == t0`,
which is only true when `t0 == 0` *or* when `t0 % store_step == t0`.  For the
typical `t0 = 0` it's fine, but documentation / type signature suggests `t0`
can be any integer.

### 4.3  `_process_idx` mutates the ComputeVar
[`base_backend.py:451-453`](pyrates/backend/base/base_backend.py#L451):

```python
if type(idx) is ComputeVar:
    idx.set_value(idx.value + self._start_idx)
    return idx.name
```

Calling `_process_idx` on the *same* ComputeVar twice on a backend with
`_start_idx=1` (julia, matlab, fortran) shifts the stored value by 2.  This is
why `JuliaBackend._process_idx` ([`julia_backend.py:260`](pyrates/backend/julia/julia_backend.py#L260))
manually flips `_start_idx` to 0 and back around the call.  Fragile.

### 4.4  `f.close()` after `with open(...)` is redundant
[`base_backend.py:360`](pyrates/backend/base/base_backend.py#L360):

```python
with open(f'{file}{self._fend}', 'w') as f:
    f.writelines(func_str)
    f.close()        # redundant — `with` already closes
```

Cosmetic, but it's a sign that the original code was rewritten incompletely.

### 4.5  `JuliaBackend.add_var_update` pops-and-rewrites the last code line
[`julia_backend.py:99-107`](pyrates/backend/julia/julia_backend.py#L99) calls
`super().add_var_update(...)`, then pops the line back off `self.code`, parses
it with `line.split(' = ')`, and re-emits with a `@.` broadcasting prefix.

`MatlabBackend.add_var_update` does the same dance ([`matlab_backend.py:87`](pyrates/backend/matlab/matlab_backend.py#L87)).

**Recommendation:** introduce a hook like
`_format_assignment(lhs: str, rhs: str, indexed: bool) -> str` that subclasses
override; remove the pop-and-rewrite.

### 4.6  Two unused imports
- `julia_backend.py:33`: `import sys` — unused.
- `matlab_backend.py:33`: `import sys` — unused.

### 4.7  `MatlabBackend` reaches up through the MRO via `super(JuliaBackend, ...)`
[`matlab_backend.py:70`](pyrates/backend/matlab/matlab_backend.py#L70) and
[`matlab_backend.py:89`](pyrates/backend/matlab/matlab_backend.py#L89) both call
`super(JuliaBackend, self).XXX` to skip the Julia layer.  This is legal Python
but very brittle — if the inheritance changes (e.g. someone introduces a
`CommonForeignBackend` between Base and Julia), Matlab silently breaks.

**Recommendation:** have `MatlabBackend` inherit directly from `BaseBackend`
and explicitly re-implement the few pieces it needs from Julia (most of which
it already overrides anyway).

---

## 5. Recommended quick wins (in priority order)

1. **Add `wsum` and `interp_rows` to `torch_funcs.py` and `jax_funcs.py`** —
   prevents silent numpy fallback in population-connectivity models. ~10 lines.
2. **Move `from numpy import f2py` and `from julia.api import Julia` inside
   their respective `__init__`s** — drops import-time crashes and shaves
   startup latency. ~4 lines.
3. **Fix `fortran_backend.py`'s `subprocess.run` to use `sys.executable`,
   drop `shell=True`, and surface stderr** — turns silent compile failures
   into actionable errors. ~6 lines.
4. **Validate `solver` argument early in each backend's `_solve`** with an
   explicit `SUPPORTED_SOLVERS` class attribute — clearer error messages,
   would have caught the torch-DDE silent fallback. ~15 lines per backend.
5. **Factor decorator application into `BaseBackend._apply_decorator`** — kill
   duplicated code in three subclasses. ~10 lines net deletion.

Items 1–3 are zero-risk drop-in patches.  Items 4–5 touch every backend but
are mechanical.

Larger refactors (lax.scan-based JAX integration, MatlabBackend MRO cleanup,
JIT module caching) should be tackled in their own commits with benchmarks.
