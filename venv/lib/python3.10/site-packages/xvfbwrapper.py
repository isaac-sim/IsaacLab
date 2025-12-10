#!/usr/bin/env python3
# Corey Goldberg, 2012-2025
# License: MIT


'''Run a headless display inside X virtual framebuffer (Xvfb)'''


import os
import platform
import subprocess
import tempfile
import time

try:
    import fcntl
except ImportError:
    system = platform.system()
    raise EnvironmentError(f'xvfbwrapper is not supported on this platform: {system}')

from random import randint


class Xvfb:

    # Maximum value to use for a display. 32-bit maxint is the
    # highest Xvfb currently supports
    MAX_DISPLAY = 2147483647

    def __init__(
        self,
        width=800,
        height=680,
        colordepth=24,
        tempdir=None,
        display=None,
        environ=None,
        timeout=10,
        **kwargs,
    ):
        self.width = width
        self.height = height
        self.colordepth = colordepth
        self._tempdir = tempdir or tempfile.gettempdir()
        self._timeout = timeout
        self.new_display = display

        if environ:
            self.environ = environ
        else:
            self.environ = os.environ

        if not self.xvfb_exists():
            msg = 'Can not find Xvfb. Please install it and try again.'
            raise EnvironmentError(msg)

        self.xvfb_cmd = []
        self.extra_xvfb_args = [
            '-screen',
            '0',
            f'{self.width}x{self.height}x{self.colordepth}',
        ]

        for key, value in kwargs.items():
            self.extra_xvfb_args += [f'-{key}', value]

        if 'DISPLAY' in self.environ:
            self.orig_display_var = self.environ['DISPLAY']
        else:
            self.orig_display_var = None

        self.proc = None

    def __enter__(self):
        # type: (...) -> Xvfb
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.new_display is not None:
            if not self._get_lock_for_display(self.new_display):
                raise ValueError(f'Could not lock display :{self.new_display}')
        else:
            self.new_display = self._get_next_unused_display()
        display_var = f':{self.new_display}'
        self.xvfb_cmd = ['Xvfb', display_var] + self.extra_xvfb_args
        self.proc = subprocess.Popen(
            self.xvfb_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        start = time.time()
        while not self._local_display_exists(self.new_display):
            time.sleep(1e-3)
            if time.time() - start > self._timeout:
                self.stop()
                raise RuntimeError(f'Xvfb display did not open: {self.xvfb_cmd}')
        ret_code = self.proc.poll()
        if ret_code is None:
            self._set_display(display_var)
        else:
            self._cleanup_lock_file()
            raise RuntimeError(f'Xvfb did not start ({ret_code}): {self.xvfb_cmd}')

    def stop(self):
        try:
            if self.orig_display_var is None:
                del self.environ['DISPLAY']
            else:
                self._set_display(self.orig_display_var)
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc.wait(self._timeout)
                except OSError:
                    pass
                self.proc = None
        finally:
            self._cleanup_lock_file()

    def xvfb_exists(self):
        # type: (...) -> bool
        '''Check that Xvfb is available on PATH and is executable.'''
        paths = self.environ['PATH'].split(os.pathsep)
        return any(os.access(os.path.join(path, 'Xvfb'), os.X_OK) for path in paths)

    def _cleanup_lock_file(self):
        '''
        This should always get called if the process exits safely
        with Xvfb.stop() (whether called explicitly, or by __exit__).

        If you are ending up with /tmp/X123-lock files when Xvfb is not
        running, then Xvfb is not exiting cleanly. Always either call
        Xvfb.stop() in a finally block, or use Xvfb as a context manager
        to ensure lock files are purged.

        '''
        self._lock_display_file.close()
        try:
            os.remove(self._lock_display_file.name)
        except OSError:
            pass

    def _get_lock_for_display(self, display):
        # type: (...) -> bool
        '''
        In order to ensure multi-process safety, this method attempts
        to acquire an exclusive lock on a temporary file whose name
        contains the display number for Xvfb.
        '''
        tempfile_path = os.path.join(self._tempdir, f'.X{display}-lock')
        try:
            self._lock_display_file = open(tempfile_path, 'w')
        except PermissionError:
            return False
        else:
            try:
                fcntl.flock(self._lock_display_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            else:
                return True

    def _get_next_unused_display(self):
        # type: (...) -> int
        '''
        Randomly chooses a display number and tries to acquire a lock for this
        number. If the lock could be acquired, returns this number, otherwise
        choses a new one.
        :return: free display number
        '''
        while True:
            rand = randint(1, self.__class__.MAX_DISPLAY)
            if self._get_lock_for_display(rand):
                return rand
            else:
                continue

    def _local_display_exists(self, display):
        temp_display_file = os.path.join(self._tempdir, '.X11-unix', f'X{display}')
        return os.path.exists(temp_display_file)

    def _set_display(self, display_var):
        self.environ['DISPLAY'] = display_var
