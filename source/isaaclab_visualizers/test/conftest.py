# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H4 hypothesis: warp 1.13 changed implicit CUDA-OpenGL sync semantics.

Monkey-patches ``newton._src.viewer.viewer_gl.ViewerGL.get_frame`` to insert an
explicit ``glFinish()`` and ``warp.synchronize_device()`` between
``glReadPixels`` (PBO async readback) and ``self._wp_pbo.map()`` (CUDA-side
buffer access). If pre-rc2/1.13 the implicit sync was tighter, this restores it.

If the failing test passes with this in place, the regression is a missed
sync between GL framebuffer-to-PBO and CUDA-mapped read.

Activated by setting VIZ_H4_EXPLICIT_SYNC=1 in the environment.
"""

from __future__ import annotations

import ctypes


def _patch_viewergl_with_explicit_sync():
    try:
        import warp as wp

        from newton._src.utils.render import copy_rgb_frame_uint8
        from newton._src.viewer.gl.opengl import RendererGL
        from newton._src.viewer.viewer_gl import ViewerGL
    except ImportError:
        return

    if getattr(ViewerGL, "_h4_patched", False):
        return

    def patched_get_frame(self, target_image=None, render_ui=False):
        gl = RendererGL.gl
        w, h = self.renderer._screen_width, self.renderer._screen_height

        if self._pbo is None:
            pbo_id = (gl.GLuint * 1)()
            gl.glGenBuffers(1, pbo_id)
            self._pbo = pbo_id[0]
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, gl.GLsizeiptr(w * h * 3), None, gl.GL_STREAM_READ)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            self._wp_pbo = wp.RegisteredGLBuffer(
                gl_buffer_id=int(self._pbo),
                device=self.device,
                flags=wp.RegisteredGLBuffer.READ_ONLY,
            )
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.renderer._frame_fbo)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)

        if render_ui and self.ui:
            self.ui.begin_frame()
            self._render_ui()
            self.ui.end_frame()
            self.ui.render()

        gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # H4 patch: force an explicit GL fence + CUDA sync before mapping the
        # PBO into CUDA address space. warp 1.13 may have weakened the
        # implicit sync inside ``RegisteredGLBuffer.map``.
        gl.glFinish()
        wp.synchronize_device(self.device)

        buf = self._wp_pbo.map(dtype=wp.uint8, shape=(w * h * 3,))

        if target_image is None:
            target_image = wp.empty(shape=(h, w, 3), dtype=wp.uint8, device=self.device)

        if target_image.shape != (h, w, 3):
            raise ValueError(f"Shape of `target_image` must be ({h}, {w}, 3), got {target_image.shape}")

        wp.launch(
            copy_rgb_frame_uint8,
            dim=(w, h),
            inputs=[buf, w, h],
            outputs=[target_image],
            device=self.device,
        )
        self._wp_pbo.unmap()

        # Diagnostic: post-call sample.
        try:
            arr = target_image.numpy()
            print(f"[H4] returned shape={arr.shape}  nonzero={int((arr != 0).sum())}  first24={arr.reshape(-1)[:24].tolist()}")
        except Exception:
            pass

        return target_image

    ViewerGL.get_frame = patched_get_frame
    ViewerGL._h4_patched = True
    print("[H4] ViewerGL.get_frame patched with explicit glFinish + wp.synchronize_device")


_patch_viewergl_with_explicit_sync()
