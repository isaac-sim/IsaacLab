Fixed
^^^^^

* Fixed the :class:`~isaaclab_physx.assets.Articulation` joint, body, and tendon index resolvers to accept
  ``torch.int64`` ID tensors by converting them to Warp ``int32`` arrays, matching the environment-ID resolver
  and the Newton backend. Previously, passing a default-dtype (``int64``) torch tensor of joint, body, or tendon
  indices to a write API raised a Warp kernel dtype error on the PhysX backend while the same call succeeded on Newton.
