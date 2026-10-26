* Deferred visualizer keyboard attachment until session startup completed, preventing startup
  failures from leaving keyboard listeners attached or camera movement disabled.
* Counted demonstration success only after simulation steps, preventing paused frames from
  completing and exporting a recording prematurely.
