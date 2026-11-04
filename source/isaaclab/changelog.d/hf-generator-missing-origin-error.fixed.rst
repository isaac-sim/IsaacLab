* Fixed :func:`~isaaclab.terrains.height_field.utils.height_field_to_mesh` raising an unrelated
  ``ValueError: too many values to unpack`` when a height-field generator returns only the height field. It now
  raises a :class:`TypeError` that names the generator and the ``(height_field, origin)`` return contract, and it
  restores the terrain size it changed for the generator when generation fails.
