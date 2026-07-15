.. currentmodule:: {{ module }}


{{ name | underline }}

.. autoclass:: {{ name }}
    :members:
    :show-inheritance:
    :exclude-members: forward

{% if "forward" in members and "forward" not in inherited_members %}
.. automethod:: {{ name }}.forward
{% endif %}
