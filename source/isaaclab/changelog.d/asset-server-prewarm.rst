Changed
^^^^^^^

* Changed Isaac Lab to open the connection to a remote asset server on a background thread
  before the first asset lookup, so the DNS resolution and the TCP and TLS handshakes overlap
  the rest of startup rather than that lookup. :class:`~isaaclab.app.AppLauncher` opens it once
  the extensions are loaded, and a kitless run, which never constructs one, opens it from
  :func:`~isaaclab.app.launch_simulation` instead. A run whose configured asset root is local
  does not open a connection.
