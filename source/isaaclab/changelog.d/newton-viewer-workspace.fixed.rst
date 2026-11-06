* Fixed paused simulations ignoring episode reset requests and continuing physics after their last window closed.
* Fixed headless simulation loops continuing after :meth:`~isaaclab.sim.SimulationContext.stop`.
* Fixed paused simulations leaving other visualizer windows unresponsive to transport controls.
