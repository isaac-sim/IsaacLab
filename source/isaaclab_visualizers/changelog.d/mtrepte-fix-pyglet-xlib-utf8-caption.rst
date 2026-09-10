Fixed
^^^^^

* Fixed ``pyglet.window.xlib.XlibException: Could not create UTF8 text property`` crashing
  the ``newton_gl``/``newton_rtx`` visualizers on some Linux environments (observed with a
  conda installation on DGX Spark). The failure came from Xlib's ``Xutf8TextListToTextProperty``
  being unable to reach the X locale database from that environment's bundled X11 client
  libraries. Newton's window captions are plain ASCII and never need UTF-8 encoding, so
  pyglet's Xlib backend is now forced onto its ASCII-only text-property codepath before Newton
  creates any window.
