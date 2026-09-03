Added
^^^^^

* Added :meth:`~isaaclab_ov.renderers.OVRTXRenderer.update_gaussian_splat_transforms` and
  :meth:`~isaaclab_ov.renderers.OVRTXRenderer.update_gaussian_splat_particles`, which play back rigid and
  deformable animated Gaussian-splat tracks on the OVRTX renderer. Both take Warp arrays and write them
  through a persistent binding without copying them, so a device-resident caller never touches the host.
  The write is asynchronous: the caller's buffer must stay unmodified until the following write to the same
  attribute, which is where the previous write is awaited.

Changed
^^^^^^^

* Changed the ovstage attribute writes to describe warp arrays where they live instead of copying them to the
  host first. The transform, geometry, camera and Gaussian-particle writes previously round-tripped through
  numpy because ``ovstage.make_dltensor`` accepts the lane-folded dtype override that ovstage columns require
  only for host arrays; the tensors are now built directly from the array pointer and submitted on the array's
  warp stream, which also removes the per-frame ``wp.synchronize_device`` those copies needed. Rendered output
  is unchanged.

* Changed the OVRTX render product to wait for ``AllLoadingFinished`` on every frame instead of only on the
  first one. Geometry streamed after the first frame, such as the geometry re-streamed when a deformable
  Gaussian track writes its per-particle arrays, was previously missing from every frame that rendered
  while the load was in flight. Frames now block while geometry is streaming.
