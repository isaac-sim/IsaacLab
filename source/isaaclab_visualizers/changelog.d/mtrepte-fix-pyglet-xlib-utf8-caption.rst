Fixed
^^^^^

* Fixed ``pyglet.window.xlib.XlibException: Could not create UTF8 text property`` crashing
  the ``newton_gl``/``newton_rtx`` visualizers on some Linux environments (observed with a
  conda installation on DGX Spark). The failure came from Xlib's ``Xutf8TextListToTextProperty``
  being unable to reach the X locale database from that environment's bundled X11 client
  libraries. Newton's window captions are plain ASCII and never need UTF-8 encoding, so
  pyglet's Xlib backend now retries setting a window's caption ASCII-only if the UTF-8 attempt
  fails, rather than raising. UTF-8 remains the default attempt and X Input Context
  (XIC)/IME keyboard input is left untouched.
