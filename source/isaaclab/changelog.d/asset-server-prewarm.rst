Changed
^^^^^^^

* Changed Isaac Lab to open the connection to a remote asset server with an asynchronous request
  before normal scene construction, so the DNS resolution and the TCP and TLS handshakes overlap
  the rest of startup rather than that lookup. :class:`~isaaclab.app.AppLauncher` opens it once
  the extensions are loaded, and a kitless run, which never constructs one, opens it from
  :func:`~isaaclab.app.launch_simulation` instead. A run whose configured asset root is local
  does not open a connection. Selected asset-region routing is applied before the request, and
  a pending request is cancelled during normal launcher teardown.
