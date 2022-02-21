PyRates.model_templates
=======================

Library of definition files (templates) for a number of neurodynamic models.
All templates can be loaded via the respective frontend classes.
For example, to load a template with base class `CircuitTemplate` that is defined at `path = model_templates.neural_mass_models.qif.qif`,
use `pyrates.frontend.CircuitTemplate.from_yaml(path)`.

pyrates.model_templates module
------------------------------

.. automodule:: pyrates.model_templates
    :members:
    :undoc-members:
    :show-inheritance:

.. toctree::

    model_templates.neural_mass_models
    model_templates.oscillators

Base Operator Templates
-----------------------

.. automodule:: model_templates.base_templates
    :members:
    :undoc-members:
    :show-inheritance:
