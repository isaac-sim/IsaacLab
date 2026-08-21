Changed
^^^^^^^

* Changed :class:`~isaaclab.app.AppLauncher` to open the connection to a remote asset server on
  a background thread once the extensions are loaded, so the DNS resolution and the TCP and TLS
  handshakes overlap the rest of startup rather than the first asset lookup. A run whose
  configured asset root is local does not open a connection.
