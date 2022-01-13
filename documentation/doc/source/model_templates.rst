PyRates.model_templates
=======================

Library of definition files (templates) for a number of neurodynamic models.
All templates can be loaded via the respective frontend classes.
For example, to load a template with base class `CircuitTemplate` that is defined at `path = model_templates.neural_mass_models.qif.qif`,
use `pyrates.frontend.CircuitTemplate.from_yaml(path)`.

Jansen-Rit Templates
--------------------

.. automodule:: model_templates.neural_mass_models.jansenrit
    :members:
    :undoc-members:
    :show-inheritance:

QIF Templates
-------------

.. automodule:: model_templates.neural_mass_models.qif
    :members:
    :undoc-members:
    :show-inheritance:

Wilson-Cowan Templates
----------------------

.. automodule:: model_templates.neural_mass_models.wilsoncowan
    :members:
    :undoc-members:
    :show-inheritance:

Kuramoto Templates
------------------

.. automodule:: model_templates.coupled_oscillators.kuramoto
    :members:
    :undoc-members:
    :show-inheritance:

Stuart-Landau Templates
-----------------------

.. automodule:: model_templates.coupled_oscillators.stuartlandau
    :members:
    :undoc-members:
    :show-inheritance:

Base Operator Templates
-----------------------

.. automodule:: model_templates.base_templates
    :members:
    :undoc-members:
    :show-inheritance:
