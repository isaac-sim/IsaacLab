The `kit-physx-isaac_rtx-distance_to_image_plane.png` golden encodes depth over a
fixed 0–100 m range. Values beyond 100 m saturate; missing depths remain black.
The runner uses the same scale, independent of each frame's maximum sample.

This baseline was regenerated from the raw Valiant depth capture collected during
the #8968 investigation with the original Isaac Sim image, manifest digest
`sha256:ccaf2879b29f5b14d3acb0f9da74c3d3305dc4f9d544a28f3278b514c742db16`.
The capture followed the two Franka soft-body cases and the Valiant RGB case.
Its median depth was 4.3707 m and its maximum was 87.1392 m.

The existing 40% pixel-difference limit is unchanged. Saved captures from the
updated renderer, including one with a 158.4695 m far sample, differ by about 22%
under this scale. Adding an incorrect 10 m offset to the original depths fails
at about 97% differing pixels.
