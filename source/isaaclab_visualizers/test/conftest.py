# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H6 fix attempt: make Newton ViewerGL's GL context current before get_frame().

Newton's ViewerGL.get_frame issues GL calls (glBindFramebuffer, glReadPixels,
RegisteredGLBuffer.map) without first calling self._make_current(). If Kit's
GLFW window context is current at the moment get_frame() runs, those GL calls
operate on Kit's context where Newton's FBO/PBO ids are invalid. The result
silently reads from Kit's framebuffer (empty for the visualizer test) or
returns zeros from glReadPixels.

This patch wraps ViewerGL.get_frame to call self.renderer._make_current()
at the start. If the visualizer test passes with this in place, the fix is
to push self._make_current() into Newton upstream's get_frame.
"""

from __future__ import annotations


def _patch_viewergl_make_current_before_get_frame():
    try:
        from newton._src.viewer.viewer_gl import ViewerGL
    except ImportError:
        return

    if getattr(ViewerGL, "_h6_patched", False):
        return

    original_get_frame = ViewerGL.get_frame

    def patched_get_frame(self, target_image=None, render_ui=False):
        try:
            self.renderer._make_current()
        except Exception as exc:  # noqa: BLE001
            print(f"[H6] _make_current failed: {exc}", flush=True)
        return original_get_frame(self, target_image=target_image, render_ui=render_ui)

    ViewerGL.get_frame = patched_get_frame
    ViewerGL._h6_patched = True
    print("[H6] ViewerGL.get_frame wrapped with _make_current()", flush=True)


_patch_viewergl_make_current_before_get_frame()
