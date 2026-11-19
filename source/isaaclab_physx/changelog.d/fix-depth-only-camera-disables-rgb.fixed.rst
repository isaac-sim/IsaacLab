* Fixed headless Isaac RTX color cameras rendering black beside depth-only cameras, including
  mixed RGB and simple-shading requests, by tracking active color render products across renderer instances.
  Restored the depth-only optimization after the last color product is cleaned up without retaining
  color demand from failed camera initialization.
