Added
^^^^^

* Added a best-effort install step (``isaaclab.sh -i``) that writes XDG ``.desktop``
  entries for the Newton GL and RTX viewer windows on Linux graphical sessions. Desktop
  environments such as GNOME resolve dock/Alt-Tab icons via a window's ``WM_CLASS``
  matched against an installed ``.desktop`` file rather than the window's own icon hint,
  so without this the dock/Alt-Tab showed a generic icon even though the window itself
  reported the correct one. No-ops on headless, CI, Docker, or Windows environments.
