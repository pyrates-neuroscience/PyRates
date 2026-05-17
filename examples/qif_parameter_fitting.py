"""
QIF mean-field parameter fitting with analytical gradient via sensitivity equations.

Demonstrates CircuitTemplate.get_jacobian_func() used to propagate parameter
gradients through an ODE trajectory with the forward sensitivity method.

Problem
-------
Recover the background excitability eta and the recurrent coupling weight J of
the Montbrió-Pazó-Roxin (2015) QIF mean-field model from a synthetic firing-rate
time series.  The model is 2-D (firing rate r, mean membrane potential v):

    tau * r' = Delta/pi + 2*r*v
    tau * v' = v^2 + eta + J*tau*r - (pi*tau*r)^2

with tau=1, Delta=1 fixed.

Gradient method
---------------
Forward sensitivity equations augment the ODE with one 2-vector per free
parameter (eta, J):

    ds_k / dt = J_ode(t, y) @ s_k + df/d_theta_k,   s_k(0) = 0

where J_ode = d f / d y is the symbolic ODE Jacobian returned by
get_jacobian_func().  The parameter sensitivities follow from

    df/d_eta = [0, 1/tau],    df/d_J = [0, r]

Integrating the 6-D augmented system yields the exact gradient of the
RMSE-over-firing-rate loss in a single forward pass.  SLSQP then uses
this gradient directly via jac=True.

Run
---
    conda run -n sbi python examples/qif_parameter_fitting.py
"""

import sys
import os
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyrates import CircuitTemplate, OperatorTemplate
from pyrates.ir.node import clear_ir_caches
from pyrates.ir.circuit import in_edge_indices, in_edge_vars
from pyrates.frontend.template import template_cache


# ─── helpers ──────────────────────────────────────────────────────────────────

def _clear_all():
    OperatorTemplate.cache.clear()
    clear_ir_caches()
    in_edge_indices.clear()
    in_edge_vars.clear()
    template_cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Compile QIF model — done exactly once
# ═══════════════════════════════════════════════════════════════════════════════
print("Compiling QIF run function …")
_clear_all()
circuit = CircuitTemplate.from_yaml("model_templates.neural_mass_models.qif.qif")
run_func, run_args, run_arg_names, sv_idx = circuit.get_run_func(
    func_name="qif_run",
    step_size=1e-3,
    solver="scipy",
    vectorize=False,
    in_place=False,
    clear=False,
)

print("Compiling QIF Jacobian function …")
_clear_all()
circuit_jac = CircuitTemplate.from_yaml("model_templates.neural_mass_models.qif.qif")
jac_func, jac_args, jac_arg_names, _ = circuit_jac.get_jacobian_func(
    func_name="qif_jac",
    step_size=1e-3,
    solver="scipy",
    vectorize=False,
    in_place=False,
    clear=True,
)

# Extract fixed constants (tau, Delta, I_ext) from compiled args
# run_arg_names: ('t','y','dy','p/qif_op/tau','p/qif_op/Delta','p/qif_op/I_ext',
#                 'p/qif_op/eta','p/in_edge_0/weight')
TAU   = float(run_args[run_arg_names.index("p/qif_op/tau")])
DELTA = float(run_args[run_arg_names.index("p/qif_op/Delta")])
I_EXT = float(run_args[run_arg_names.index("p/qif_op/I_ext")])
Y0    = np.array([0.8, -0.5])                 # initial state in the active-r regime

# jac_arg_names: ('t','y','p/in_edge_0/weight','p/qif_op/tau','p/qif_op/Delta',
#                 'p/qif_op/I_ext','p/qif_op/eta')
print(f"  Fixed constants:  tau={TAU}, Delta={DELTA}, I_ext={I_EXT}")
print(f"  Initial state:    r0={Y0[0]}, v0={Y0[1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Thin wrappers — route calls to compiled functions
# ═══════════════════════════════════════════════════════════════════════════════

def qif_rhs(t: float, y: np.ndarray, eta: float, J: float) -> np.ndarray:
    """Return f(y; eta, J) — the QIF vector field at (t, y)."""
    dy = np.zeros(2, dtype=np.float32)
    return np.asarray(
        run_func(
            np.float32(t), y.astype(np.float32), dy,
            np.float32(TAU), np.float32(DELTA), np.float32(I_EXT),
            np.float32(eta), np.float32(J),
        ),
        dtype=float,
    )


def qif_jacobian(t: float, y: np.ndarray, eta: float, J: float) -> np.ndarray:
    """Return J_ode = df/dy from the symbolic Jacobian function.

    This is the 2×2 matrix produced by get_jacobian_func():
        J[0,0] = 2v/tau,       J[0,1] = 2r/tau
        J[1,0] = J - 2pi^2*r,  J[1,1] = 2v/tau   (tau=1)
    """
    # jac_func signature: (t, y, weight, tau, Delta, I_ext, eta)
    return np.asarray(
        jac_func(
            np.float32(t), y.astype(np.float32),
            np.float32(J), np.float32(TAU), np.float32(DELTA), np.float32(I_EXT),
            np.float32(eta),
        ),
        dtype=float,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Generate synthetic target data
# ═══════════════════════════════════════════════════════════════════════════════
T       = 30.0                            # ms — captures the informative transient
T_EVAL  = np.linspace(0.0, T, 300)       # 300 sample points

ETA_TARGET, J_TARGET = -2.0, 15.0        # ground-truth parameters
ETA_INIT,  J_INIT   = -3.5, 10.0        # starting guess (different basin)

print(f"\nTarget parameters:  eta={ETA_TARGET}, J={J_TARGET}")
print(f"Initial guess:      eta={ETA_INIT},  J={J_INIT}")

sol_target = solve_ivp(
    qif_rhs, (0.0, T), Y0.copy(),
    args=(ETA_TARGET, J_TARGET),
    t_eval=T_EVAL, method="RK45", max_step=0.5, rtol=1e-6, atol=1e-9,
)
r_target = sol_target.y[0]              # only firing rate is observed


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Augmented ODE: state + forward sensitivities
# ═══════════════════════════════════════════════════════════════════════════════
# State vector: y_aug = [r, v,  s_eta_r, s_eta_v,  s_J_r, s_J_v]
# s_eta = dy/d(eta),  s_J = dy/d(J)
#
# Sensitivity equations (derived from differentiating the QIF ODE wrt theta):
#   ds_k/dt = J_ode(y) @ s_k + df/d_theta_k
#
# Analytical parameter Jacobians (from v' = (v² + eta + J*tau*r - ...)/tau):
#   df/d_eta = [0,     1/tau]   (v equation has +eta/tau)
#   df/d_J   = [0,     r    ]   (v equation has +J*tau*r/tau = J*r)

DF_DETA = np.array([0.0, 1.0 / TAU])    # constant

def augmented_rhs(t: float, y_aug: np.ndarray, eta: float, J: float) -> np.ndarray:
    y     = y_aug[:2]
    s_eta = y_aug[2:4]
    s_J   = y_aug[4:6]

    dy    = qif_rhs(t, y, eta, J)
    J_ode = qif_jacobian(t, y, eta, J)   # 2×2 ODE Jacobian from get_jacobian_func

    df_dJ = np.array([0.0, y[0]])        # r = y[0]; depends on current state

    ds_eta = J_ode @ s_eta + DF_DETA
    ds_J   = J_ode @ s_J   + df_dJ

    return np.concatenate([dy, ds_eta, ds_J])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Loss function + analytical gradient (returned together for SLSQP jac=True)
# ═══════════════════════════════════════════════════════════════════════════════

eval_count = [0]

def loss_and_grad(params):
    eta, J = float(params[0]), float(params[1])
    eval_count[0] += 1

    y_aug0 = np.zeros(6)
    y_aug0[:2] = Y0

    sol = solve_ivp(
        augmented_rhs, (0.0, T), y_aug0,
        args=(eta, J),
        t_eval=T_EVAL, method="RK45", max_step=0.5, rtol=1e-6, atol=1e-9,
    )

    r_sim   = sol.y[0]           # simulated firing rate
    s_eta_r = sol.y[2]           # dr/d(eta)
    s_J_r   = sol.y[4]           # dr/d(J)

    diff = r_sim - r_target
    N    = len(diff)
    mse  = np.mean(diff ** 2)
    rmse = float(np.sqrt(mse))

    if rmse > 1e-12:
        # chain rule: dRMSE/d_theta_k = mean[(r-r*) * s_k_r] / rmse
        dL_deta = float(np.mean(diff * s_eta_r)) / rmse
        dL_dJ   = float(np.mean(diff * s_J_r))   / rmse
    else:
        dL_deta = dL_dJ = 0.0

    return rmse, np.array([dL_deta, dL_dJ])


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Validate gradient against finite differences at the initial point
# ═══════════════════════════════════════════════════════════════════════════════
print("\nValidating analytical gradient against finite differences …")
eps = 1e-4
L0, g_ana = loss_and_grad(np.array([ETA_INIT, J_INIT]))
L_ep, _ = loss_and_grad(np.array([ETA_INIT + eps, J_INIT]))
L_ej, _ = loss_and_grad(np.array([ETA_INIT, J_INIT + eps]))
g_fd = np.array([(L_ep - L0) / eps, (L_ej - L0) / eps])

print(f"  L(x0) = {L0:.6f}")
print(f"  Analytical gradient:  dL/deta={g_ana[0]:.6f},  dL/dJ={g_ana[1]:.6f}")
print(f"  Finite-diff gradient: dL/deta={g_fd[0]:.6f},   dL/dJ={g_fd[1]:.6f}")
rel_err = np.linalg.norm(g_ana - g_fd) / (np.linalg.norm(g_fd) + 1e-12)
print(f"  Relative error: {rel_err:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SLSQP optimisation
# ═══════════════════════════════════════════════════════════════════════════════
param_history = [np.array([ETA_INIT, J_INIT])]

def callback(xk):
    param_history.append(xk.copy())

print(f"\nRunning SLSQP optimisation …")
result = minimize(
    loss_and_grad,
    x0=np.array([ETA_INIT, J_INIT]),
    method="SLSQP",
    jac=True,                           # loss_and_grad returns (value, gradient)
    bounds=[(-7.0, 0.0), (4.0, 22.0)],
    callback=callback,
    options={"maxiter": 80, "ftol": 1e-10, "disp": False},
)

eta_fit, J_fit = result.x
print(f"\nOptimisation finished in {result.nit} iterations "
      f"({eval_count[0]} function evaluations)")
print(f"  Target:    eta={ETA_TARGET:.4f},  J={J_TARGET:.4f}")
print(f"  Recovered: eta={eta_fit:.4f},  J={J_fit:.4f}")
print(f"  Final RMSE: {result.fun:.2e}")
print(f"  Success: {result.success}  |  {result.message}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Plot results
# ═══════════════════════════════════════════════════════════════════════════════
sol_init = solve_ivp(
    qif_rhs, (0.0, T), Y0.copy(),
    args=(ETA_INIT, J_INIT),
    t_eval=T_EVAL, method="RK45", max_step=0.5,
)
sol_fit = solve_ivp(
    qif_rhs, (0.0, T), Y0.copy(),
    args=(eta_fit, J_fit),
    t_eval=T_EVAL, method="RK45", max_step=0.5,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: firing-rate time series
ax = axes[0]
ax.plot(T_EVAL, r_target,     color="C0", lw=2,   label=f"target   (η={ETA_TARGET}, J={J_TARGET})")
ax.plot(T_EVAL, sol_init.y[0], color="C3", lw=1.5, ls="--", label=f"initial  (η={ETA_INIT}, J={J_INIT})")
ax.plot(T_EVAL, sol_fit.y[0],  color="C2", lw=1.5, ls=":",  label=f"fitted   (η={eta_fit:.2f}, J={J_fit:.2f})")
ax.set_xlabel("time (ms)")
ax.set_ylabel("firing rate r")
ax.set_title("QIF firing-rate dynamics")
ax.legend(fontsize=8)

# Panel 2: parameter trajectory during optimisation
param_arr = np.array(param_history)
ax = axes[1]
sc = ax.scatter(param_arr[:, 0], param_arr[:, 1],
                c=np.arange(len(param_arr)), cmap="plasma",
                s=40, zorder=3)
ax.plot(param_arr[:, 0], param_arr[:, 1], "k-", lw=0.8, alpha=0.5)
ax.scatter([ETA_TARGET], [J_TARGET], marker="*", s=200, color="C0",
           zorder=5, label="target")
ax.scatter([ETA_INIT],   [J_INIT],   marker="o", s=80,  color="C3",
           zorder=5, label="initial")
ax.scatter([eta_fit],    [J_fit],    marker="s", s=80,  color="C2",
           zorder=5, label="fitted")
plt.colorbar(sc, ax=ax, label="iteration")
ax.set_xlabel("eta")
ax.set_ylabel("J")
ax.set_title("Optimisation trajectory")
ax.legend(fontsize=8)

# Panel 3: gradient validation — analytical vs finite-difference
ax = axes[2]
etas_scan = np.linspace(-8.0, 0.5, 15)
g_ana_eta, g_fd_eta = [], []
for e in etas_scan:
    L, ga = loss_and_grad(np.array([e, J_TARGET]))
    Lp, _ = loss_and_grad(np.array([e + eps, J_TARGET]))
    g_ana_eta.append(ga[0])
    g_fd_eta.append((Lp - L) / eps)

ax.plot(etas_scan, g_ana_eta, "C0-o", ms=4, label="analytical (Jacobian)")
ax.plot(etas_scan, g_fd_eta,  "C3--s", ms=4, label="finite-difference")
ax.axvline(ETA_TARGET, color="gray", lw=1, ls=":")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("eta")
ax.set_ylabel("dRMSE / d_eta")
ax.set_title("Gradient validation  (J = J_target)")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "qif_parameter_fitting.png")
plt.savefig(out_path, dpi=120)
print(f"\nFigure saved to {out_path}")
