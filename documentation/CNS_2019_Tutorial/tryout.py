from pyrates.frontend import CircuitTemplate, NodeTemplate, OperatorTemplate
from pyrates.utility import plot_timeseries, grid_search, create_cmap

import numpy as np
import matplotlib.pyplot as plt

Js = np.linspace(0, 100, 10)
T = 5.
dt = 1e-4
dts = 1e-3
ext_input = np.random.uniform(3., 5., (int(T/dt), 1))

exc_syn = ['d/dt * r = (delta/(PI*tau) + 2.*r*v)/tau',
           'd/dt * v = (v^2 + eta + I_ext + J*I_syn*tau - (PI*r*tau)^2)/tau',
           'd/dt * I_syn = x/tau_syn',
           'd/dt * x = (r+r_exc-r_inh - 2.*x - I_syn)/tau_syn']
inh_syn = ['d/dt * r = (delta/(PI*tau) + 2.*r*v)/tau',
           'd/dt * v = (v^2 + eta + I_ext + J*I_syn*tau - (PI*r*tau)^2)/tau',
           'd/dt * I_syn = x/tau_syn',
           'd/dt * x = (r_exc-r-r_inh - 2.*x - I_syn)/tau_syn']
variables = {'delta': {'default': 1.0},
             'tau': {'default': 1.0},
             'eta': {'default': -5.0},
             'J': {'default': 15.0},
             'tau_syn': {'default': 2.0},
             'r': {'default': 'variable'},
             'v': {'default': 'variable'},
             'I_syn': {'default': 'variable'},
             'x': {'default': 'variable'},
             'I_ext': {'default': 'input'},
             'r_exc': {'default': 'input'},
             'r_inh': {'default': 'input'},
             }

op_exc_syn = OperatorTemplate(name='qif_exc_syn', path=None, equations=exc_syn, variables=variables)
op_inh_syn = OperatorTemplate(name='qif_inh_syn', path=None, equations=inh_syn, variables=variables)

pc = NodeTemplate(name='pc', path=None, operators=[op_exc_syn])
ein = NodeTemplate(name='ein', path=None, operators={op_exc_syn: {'eta': -2.5}})
iin = NodeTemplate(name='iin', path=None, operators={op_inh_syn: {'eta': 0.5, 'tau': 4.0}})

circuit_template = CircuitTemplate(name='circuit', path=None,
                                   nodes={'pc': pc, 'ein': ein, 'iin': iin},
                                   edges=[('pc/qif_exc_syn.0/r', 'ein/qif_exc_syn.1/r_exc',
                                           None, {'weight': 1.}),
                                          ('ein/qif_exc_syn.1/r', 'pc/qif_exc_syn.0/r_exc',
                                           None, {'weight': 1.}),
                                          ('pc/qif_exc_syn.0/r', 'iin/qif_inh_syn.1/r_exc',
                                           None, {'weight': 1.}),
                                          ('iin/qif_inh_syn.1/r', 'pc/qif_exc_syn.0/r_inh',
                                           None, {'weight': 1.})]
                                   )

results = grid_search(circuit_template, param_grid={'J': Js},
                      param_map={'J': {'var': [('qif_exc_syn.0', 'J'),
                                               ('qif_exc_syn.1', 'J'),
                                               ('qif_inh_syn.1', 'J')],
                                       'nodes': ['pc.0', 'ein.0', 'iin.0']}},
                      simulation_time=T, dt=dt, sampling_step_size=dts,
                      inputs={('pc.0', 'qif_exc_syn.0', 'I_ext'): ext_input},
                      outputs={'r_PC': ('pc.0', 'qif_exc_syn.0', 'r')},
                      init_kwargs={'vectorization': 'nodes', 'build_in_place': False},
                      permute_grid=False)

fig, axes = plt.subplots(nrows=len(Js), figsize=(10, 20))
cmap = create_cmap('pyrates_purple', as_cmap=False, n_colors=1, reverse=True)
for i, ax in enumerate(axes):
    plot_timeseries(results.iloc[results.index > 1.0, i], ax=ax, cmap=cmap,
                    ylabel='r')
    ax.legend([f'J = {results.columns.values[i][-1]}'], loc='upper right')
plt.show()
