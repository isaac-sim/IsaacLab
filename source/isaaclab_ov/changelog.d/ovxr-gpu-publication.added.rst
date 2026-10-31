* Added explicit rendering GPU selection and completed geometry copies for applications
  using separate physics and rendering devices.
* Published rigid transforms and deformable points through completed target-device
  buffers. Failed transfers retained their source storage until the producing and
  transfer streams drained successfully, before releasing the native-owner lease guards.
