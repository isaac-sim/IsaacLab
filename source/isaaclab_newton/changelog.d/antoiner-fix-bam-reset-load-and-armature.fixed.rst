* Fixed the first BAM friction budget after an environment reset using the terminated episode's external
  load, by clearing the previous solve's bias and constraint forces of the reset worlds in the MJWarp reset hook.
