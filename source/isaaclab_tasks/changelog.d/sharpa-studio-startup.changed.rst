* Made the coffee and teapot demos use task-only studio backdrops in Newton GL by default, with full scenery available through ``--backdrop`` and retained by default in Kit and OVRTX.
* Limited the RJ45 demo's default Newton GL view to the detailed gold switch front and all RJ45 ports, with one full rack available through ``--backdrop datacenter`` and the full data center retained in Kit and OVRTX.
* Prepared the RJ45 switch-front meshes once so Newton GL loads their geometry and materials without extracting or cropping the rack at startup.
* Skipped visual-only Newton shape imports when the Sharpa demos use Kit or OVRTX without Newton GL.
* Deferred the teapot demo's large geometry and particle list assignments until after scene config construction to avoid repeated deep copies.
