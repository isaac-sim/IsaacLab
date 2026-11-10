* Fixed headless Isaac RTX color cameras rendering black when a depth-only camera exists in the same process,
  by enabling the process-wide ``/rtx/sdg/force/disableColorRender`` only while no camera renders color.
