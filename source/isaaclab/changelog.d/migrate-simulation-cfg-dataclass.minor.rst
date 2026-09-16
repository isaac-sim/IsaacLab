Changed
^^^^^^^

* Converted ``SimulationCfg`` to a standard dataclass while preserving its configuration helper methods. Downstream
  subclasses may use ``@dataclass`` directly; existing ``@configclass`` subclasses remain supported.
