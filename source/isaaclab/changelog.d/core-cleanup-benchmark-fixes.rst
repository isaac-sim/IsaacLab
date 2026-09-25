Fixed
^^^^^

* Fixed :meth:`~isaaclab.benchmark.measurements.TestPhase.from_json` raising ``TypeError`` for phases
  serialized with :class:`~isaaclab.benchmark.measurements.TestPhaseEncoder` that have measurements, and
  dropping the metadata of phases without measurements. This also affected
  :meth:`~isaaclab.benchmark.measurements.TestPhase.aggregate_json_files`.
* Fixed :meth:`~isaaclab.benchmark.recorders.record_cpu_info.CPUInfoRecorder.get_data` raising ``KeyError``
  when called before the first :meth:`~isaaclab.benchmark.recorders.record_cpu_info.CPUInfoRecorder.update`.
