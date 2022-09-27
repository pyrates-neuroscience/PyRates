*******************************************
Example: Two rate-coupled leaky integrators
*******************************************

This example is a minimal example for a numerical simulation performed in `PyRates`,
based on a pre-implemented model.
It is taken from a `jupyter notebook <https://github.com/pyrates-neuroscience/PyRates/blob/master/documentation/Tutorial_PyRates_Basics.ipynb>`_, which provides detailed insight
into the model and how to implement it and use it in `PyRates`.

The following code performs a numerical integration of the evolution equations of the Jansen-Rit neural mass model
over a time interval of 2 s.

.. code-block::

    from pyrates import integrate

    # model definition
    model_def = "model_templates.neural_mass_models.jansenrit.JRC"

    # simulation
    results = integrate(model_def, simulation_time=2.0, step_size=1e-4,
                        outputs={'psp_e': 'pc/rpo_e_in/V',
                                 'psp_i': 'pc/rpo_i/V'},
                        clear=True)

The variable :code:`model_def` is a pointer to the model definition and the call to :code:`integrate`
performs the numerical integration and returns a :code:`pandas.DataFrame` with the resulting time series.

See the `The Jansen-Rit Neural Mass Model <https://pyrates.readthedocs.io/en/latest/auto_introductions/jansenrit.html>`_ use example for a more detailed explanation of the Jansen-Rit model, the
`jupyter notebook <https://github.com/pyrates-neuroscience/PyRates/blob/master/documentation/Tutorial_PyRates_Basics.ipynb>`_ for details on how to implement this model by your self, and the `Numerical Simulations <https://pyrates.readthedocs.io/en/latest/auto_analysis/simulations.html>`_ use example for details on the numerical integration
capacities of `PyRates`.
