Added
^^^^^

* ``isaaclab --editor`` now also generates a ``.desktop`` file at
  ``~/.local/share/applications/isaaclab.desktop`` on Linux, fixing the Kit visualizer window
  showing a generic taskbar/dock icon instead of the Isaac Sim icon. Most Linux desktop
  environments resolve taskbar icons by matching a running window's ``WM_CLASS`` against an
  installed ``.desktop`` file's ``StartupWMClass``, not from the window's own ``_NET_WM_ICON``
  hint (which Kit already sets correctly). The generated ``StartupWMClass`` is read from
  ``apps/isaaclab.python.kit``, matching what Kit composes for the running window.
