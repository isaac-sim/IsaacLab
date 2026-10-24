* Fixed Newton contact sensors with identical body and filter expressions reporting zero forces for all but the
  last one created. Each new sensor had replaced the Newton sensor of the earlier ones, which now share it.
