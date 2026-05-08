# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic instrumentation for ViewerGL.get_frame.

Logs context state and the PBO/CUDA-GL interop result so we can tell whether
the FBO itself is empty (rendering didn't happen) or whether the PBO/CUDA
readback path returned zeros while the FBO had content.

The instrumentation is installed via pytest_collection_modifyitems so Newton
is imported AFTER the test file has run AppLauncher (which sets up Kit's pxr
bindings). Importing Newton at conftest top-level pulls pxr in too early and
causes a TypeError on Kit 110+ ("No to_python converter for GfVec3f").
"""

from __future__ import annotations

import ctypes


def _gl_err_str(code):
    names = {
        0: "GL_NO_ERROR",
        0x0500: "GL_INVALID_ENUM",
        0x0501: "GL_INVALID_VALUE",
        0x0502: "GL_INVALID_OPERATION",
        0x0503: "GL_STACK_OVERFLOW",
        0x0504: "GL_STACK_UNDERFLOW",
        0x0505: "GL_OUT_OF_MEMORY",
        0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION",
    }
    return names.get(int(code), f"0x{int(code):04x}")


def _log_environment_once():
    try:
        import sys

        import newton as nt
        import warp as wp

        print(f"[VIZDIAG] python={sys.version.split()[0]}", flush=True)
        print(f"[VIZDIAG] warp={wp.__version__}  newton={nt.__version__}", flush=True)
        try:
            import pyglet

            print(f"[VIZDIAG] pyglet={pyglet.version}  headless={pyglet.options.get('headless')}", flush=True)
        except Exception as exc:
            print(f"[VIZDIAG] pyglet info error: {exc}", flush=True)
    except Exception as exc:
        print(f"[VIZDIAG] env-log error: {exc}", flush=True)


def _instrument_viewergl_once():
    try:
        from newton._src.viewer.viewer_gl import ViewerGL
    except ImportError:
        return

    if getattr(ViewerGL, "_vizdiag_instrumented", False):
        return

    original_get_frame = ViewerGL.get_frame

    def instrumented_get_frame(self, target_image=None, render_ui=False):
        import numpy as np  # noqa: PLC0415

        from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

        gl = RendererGL.gl
        prefix = "[VIZDIAG]"

        if not getattr(ViewerGL, "_vizdiag_env_logged", False):
            _log_environment_once()
            ViewerGL._vizdiag_env_logged = True

        try:
            fbo = self.renderer._frame_fbo
            w, h = self.renderer._screen_width, self.renderer._screen_height
        except Exception as exc:
            print(f"{prefix} renderer state error: {exc}", flush=True)
            return original_get_frame(self, target_image=target_image, render_ui=render_ui)

        print(f"{prefix} fbo={fbo!r}  pbo={self._pbo!r}  wp_pbo={self._wp_pbo!r}  size={w}x{h}", flush=True)
        e0 = gl.glGetError()
        print(f"{prefix} glGetError before: {_gl_err_str(e0)}", flush=True)

        # Direct CPU readback of the FBO (independent of the PBO/CUDA path).
        try:
            assert fbo is not None
            cpu_buf = (ctypes.c_uint8 * (w * h * 3))()
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, cpu_buf)
            cpu_err = gl.glGetError()
            cpu_arr = np.frombuffer(cpu_buf, dtype=np.uint8)
            cpu_nonzero = int((cpu_arr != 0).sum())
            cpu_max = int(cpu_arr.max()) if cpu_arr.size else 0
            print(
                f"{prefix} CPU-readback: nonzero={cpu_nonzero}/{w * h * 3}  max={cpu_max}  "
                f"first12={cpu_arr[:12].tolist()}  err={_gl_err_str(cpu_err)}",
                flush=True,
            )
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        except Exception as exc:
            print(f"{prefix} CPU-readback error: {exc}", flush=True)

        result = original_get_frame(self, target_image=target_image, render_ui=render_ui)
        e1 = gl.glGetError()
        print(f"{prefix} glGetError after: {_gl_err_str(e1)}", flush=True)

        try:
            arr = result.numpy()
            nonzero = int((arr != 0).sum())
            print(
                f"{prefix} PBO-result: shape={arr.shape}  nonzero={nonzero}/{arr.size}  "
                f"max={int(arr.max()) if arr.size else 0}  first12={arr.reshape(-1)[:12].tolist()}",
                flush=True,
            )
        except Exception as exc:
            print(f"{prefix} PBO-result sample error: {exc}", flush=True)

        return result

    ViewerGL.get_frame = instrumented_get_frame
    ViewerGL._vizdiag_instrumented = True
    print("[VIZDIAG] ViewerGL.get_frame instrumented", flush=True)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Hook fires after test files are collected (i.e., after AppLauncher in test files runs)."""
    _instrument_viewergl_once()
