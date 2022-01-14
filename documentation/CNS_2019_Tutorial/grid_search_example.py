from pyrates.frontend import OperatorTemplate, NodeTemplate, CircuitTemplate
from pyrates.utility_old.grid_search import grid_search
from pyrates.utility_old.visualization import Interactive2DParamPlot, plot_psd
import numpy as np
import matplotlib.pyplot as plt

exc_syn = ['d/dt * r = (delta/(PI*tau) + 2.*r*v) /tau',
           'd/dt * v = (v^2 + eta + I_ext + (I_exc - I_inh)*tau - (PI*r*tau)^2) /tau',
           'd/dt * I_exc = J*r + r_exc - I_exc/tau_exc',
           'd/dt * I_inh =  r_inh - I_inh/tau_inh'
          ]
inh_syn = ['d/dt * r = (delta/(PI*tau) + 2.*r*v) /tau',
           'd/dt * v = (v^2 + eta + I_ext + (I_exc - I_inh)*tau - (PI*r*tau)^2) /tau',
           'd/dt * I_exc = r_exc - I_exc/tau_exc',
           'd/dt * I_inh = J*r + r_inh - I_inh/tau_inh'
          ]
variables = {'delta': {'default': 1.0},
             'tau': {'default': 1.0},
             'eta': {'default': -2.5},
             'J': {'default': 0.0},
             'tau_exc': {'default': 1.0},
             'tau_inh': {'default': 2.0},
             'r': {'default': 'output'},
             'v': {'default': 'variable'},
             'I_exc': {'default': 'variable'},
             'I_inh': {'default': 'variable'},
             'I_ext': {'default': 'input'},
             'r_exc': {'default': 'input'},
             'r_inh': {'default': 'input'},
             }

op_exc_syn = OperatorTemplate(name='Op_exc_syn', path=None, equations=exc_syn, variables=variables)
op_inh_syn = OperatorTemplate(name='Op_inh_syn', path=None, equations=inh_syn, variables=variables)

pcs = NodeTemplate(name='PCs', path=None, operators=[op_exc_syn])
eins = NodeTemplate(name='EINs', path=None, operators={op_exc_syn: {'eta': -0.5}})
iins = NodeTemplate(name='IINs', path=None, operators={op_inh_syn: {'tau': 2.0, 'eta': -0.5}})

jrc_template = CircuitTemplate(name='jrc_template', path=None,
                               nodes={'PCs': pcs, 'EINs': eins, 'IINs': iins},
                               edges=[('PCs/Op_exc_syn/r', 'EINs/Op_exc_syn/r_exc',
                                       None, {'weight': 13.5}),
                                      ('EINs/Op_exc_syn/r', 'PCs/Op_exc_syn/r_exc',
                                       None, {'weight': 0.8*13.5}),
                                      ('PCs/Op_exc_syn/r', 'IINs/Op_inh_syn/r_exc',
                                       None, {'weight': 0.25*13.5}),
                                      ('IINs/Op_inh_syn/r', 'PCs/Op_exc_syn/r_inh',
                                       None, {'weight': 1.75*13.5})]
                               )

w_ein_pc = np.linspace(0.5, 2, 10) * 0.8*13.5
w_iin_pc = np.linspace(0.5, 2, 10) * 1.75*13.5

T = 100.
dt = 1e-3
dts = 1e-2
ext_input = np.random.uniform(3., 5., (int(T/dt), 1))

results, t, _ = grid_search(jrc_template,
                            param_grid={'w_ep': w_ein_pc, 'w_ip': w_iin_pc},
                            param_map={'w_ep': {'var': [(None, 'weight')],
                                                'edges': [('EINs.0', 'PCs.0', 0)]},
                                       'w_ip': {'var': [(None, 'weight')],
                                                'edges': [('IINs.0', 'PCs.0', 0)]}},
                            simulation_time=T, dt=dt, sampling_step_size=dts,
                            inputs={('PCs.0', 'Op_exc_syn.0', 'I_ext'): ext_input},
                            outputs={'r': ('PCs.0', 'Op_exc_syn.0', 'r')},
                            init_kwargs={'backend': 'numpy', 'vectorization': 'nodes', 'build_in_place': False},
                            permute_grid=True,
                            profile='t',
                            verbose=True)

# calculate power-spectral density of firing rate fluctuations
max_freq = np.zeros((len(w_ein_pc), len(w_iin_pc)))
max_pow = np.zeros_like(max_freq)
for we in w_ein_pc:
    for wi in w_iin_pc:
        plot_psd(results[we][wi], tmin=30.0, show=False)
        p = plt.gca().get_lines()[-1].get_ydata()
        f = plt.gca().get_lines()[-1].get_xdata()
        idx_r, idx_c = np.argwhere(w_ein_pc == we)[0], np.argwhere(w_iin_pc == wi)[0]
        max_idx = np.argmax(p)
        max_freq[idx_r, idx_c] = f[max_idx]
        max_pow[idx_r, idx_c] = p[max_idx]
        plt.close(plt.gcf())

Interactive2DParamPlot(max_freq, results, w_iin_pc, w_ein_pc, tmin=30.0)
plt.show()
