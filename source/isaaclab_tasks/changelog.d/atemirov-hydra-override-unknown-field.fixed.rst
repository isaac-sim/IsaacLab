* Fixed ``env.`` and ``agent.`` command-line overrides silently adding a misspelled field to the configuration and
  leaving the intended field unchanged. An override of a field the configuration does not have now raises an
  :class:`AttributeError` that names the override and the missing field.
