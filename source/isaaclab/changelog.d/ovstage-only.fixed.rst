* Fixed camera tensor processors receiving no environment mask when an OVRTX camera set ``async_rendering=True``;
  OVRTX now renders synchronously, so processors receive the mask of the current capture.
