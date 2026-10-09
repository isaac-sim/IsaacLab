* Removed Git LFS from the base, cuRobo, and kit-less container images, because scanners flag the Go modules
  embedded in its upstream binary. ``git`` is still installed; install Git LFS in a derived image to
  work with Git LFS repositories inside a container.
