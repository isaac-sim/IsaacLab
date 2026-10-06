* Fixed scene construction on Newton 1.7 pre-releases, which record cloth and soft meshes as surface and volume
  deformable objects when they are added: the cloner no longer records them a second time and the backend reads
  the renamed builder groups.
