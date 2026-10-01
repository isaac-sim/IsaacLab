* Added :attr:`~isaaclab.visualizers.BaseVisualizer.key_event_source` and
  :class:`~isaaclab.visualizers.KeyEventSource`, keyboard input from a focused visualizer window.
  Listeners receive each key press and release as a W3C ``KeyboardEvent.code`` string and a focus
  loss when the window stops delivering keys. :meth:`~isaaclab.visualizers.KeyEventSource.add_key_listener`
  returns a :class:`~isaaclab.visualizers.KeyboardSubscription` and
  :meth:`~isaaclab.visualizers.KeyEventSource.capture_keyboard` returns a
  :class:`~isaaclab.visualizers.KeyboardCapture`, which suspends the backend's own key bindings
  until closed. :class:`~isaaclab.visualizers.KeyboardCapabilities` states whether a backend reports
  physical keys and withholds keys typed into its UI. Visualizers without a local window that can
  report keys return ``None``, the default.
