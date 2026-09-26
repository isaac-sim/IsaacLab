Changed
^^^^^^^

* Changed the Newton sensor graph to upload its task flags only when they change, removing a
  host-to-device copy and stream synchronization from every render.
