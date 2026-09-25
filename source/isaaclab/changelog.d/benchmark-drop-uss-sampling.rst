Removed
^^^^^^^

* Removed Unique Set Size (USS) collection from :class:`~isaaclab.benchmark.recorders.MemoryInfoRecorder`.
  ``psutil.Process.memory_full_info()`` walks the process page tables on every call, and the recorder ran it
  once per second from the :class:`~isaaclab.benchmark.BenchmarkMonitor` thread while the benchmark was being
  timed, perturbing the workload being measured. The ``System Memory USS``, ``System Memory USS std``,
  ``System Memory USS peak`` and ``System Memory USS n`` measurements are no longer emitted. Resident Set Size
  and Virtual Memory Size are unchanged and are read from cheap kernel counters.
