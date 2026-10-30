* Reduced coffee and teapot startup costs by keeping particle and mesh data in compact NumPy arrays.
* Skipped unused Newton visual geometry imports in the coffee and RJ45 examples when running without Newton GL.
* Stopped generating unused hydroelastic contact-surface visualization data in the RJ45 example.
* Reduced RJ45 scene preparation costs by loading prepared switch-front geometry and materials.
* Reduced RJ45 robot texture loading costs with prepared 512-pixel texture overrides that reused the existing geometry and physics.
