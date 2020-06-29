PyRates.model_templates
=======================

Library of definition files (templates) for a number of neurodynamic models.
All templates can be loaded via the respective frontend classes.
For example, to load a template with base class `CircuitTemplate` that is defined at `path = model_templates.montbrio.simple_montbrio.Net1`,
use `pyrates.frontend.CircuitTemplate.from_yaml(path)`.

Jansen-Rit Templates
--------------------

.. automodule:: model_templates.jansen_rit.simple_jansenrit
    :members:
    :undoc-members:
    :show-inheritance:

QIF Templates
-------------

.. automodule:: model_templates.montbrio.simple_montbrio
    :members:
    :undoc-members:
    :show-inheritance:

Kuramoto Templates
------------------

.. automodule:: model_templates.kuramoto.simple_kuramoto
    :members:
    :undoc-members:
    :show-inheritance:

Wilson-Cowan Templates
----------------------

.. automodule:: model_templates.kuramoto.simple_wilsoncowan
    :members:
    :undoc-members:
    :show-inheritance:
