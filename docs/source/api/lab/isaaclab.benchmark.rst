isaaclab.benchmark
==================

.. automodule:: isaaclab.benchmark

   .. rubric:: Request and result classes

   .. autosummary::

      BenchmarkLauncherConfig
      BenchmarkOutputConfig
      BenchmarkPlayRequest
      BenchmarkResult
      BenchmarkRuntimeRequest
      BenchmarkStartupRequest
      BenchmarkTrainingRequest

   .. rubric:: Workflow functions

   .. autosummary::

      run_benchmark
      run_play_benchmark
      run_runtime_benchmark
      run_startup_benchmark
      run_training_benchmark

   .. rubric:: Benchmark framework

   .. autosummary::

      BaseIsaacLabBenchmark
      BenchmarkMonitor
      MethodBenchmarkDefinition
      MethodBenchmarkRunner
      MethodBenchmarkRunnerConfig

   .. rubric:: Micro-benchmarks

   .. autosummary::

      LatencyBenchmarkRunner
      LatencySample
      LatencyStatistics
      MicrobenchmarkCommand
      MicrobenchmarkFactory
      add_latency_measurements
      measure_latency
      run_microbenchmark_cli
      summarize_latency

   .. rubric:: Measurements and metadata

   .. autosummary::

      BooleanMeasurement
      DictMeasurement
      DictMetadata
      FloatMetadata
      IntMetadata
      ListMeasurement
      Measurement
      MetadataBase
      SingleMeasurement
      StatisticalMeasurement
      StringMetadata
      TestPhase

   .. rubric:: Result schema

   .. autosummary::

      EnvironmentStepTiming
      Learning
      LearningCurve
      MeanStd
      PlayBundle
      Resources
      RunConfig
      RunIdentity
      Runtime
      RuntimeBundle
      StartupBundle
      StartupPhase
      StartupTime
      TrainingBundle


.. currentmodule:: isaaclab.benchmark

Request and Result Classes
--------------------------

.. autoclass:: BenchmarkLauncherConfig
   :members:

.. autoclass:: BenchmarkOutputConfig
   :members:

.. autoclass:: BenchmarkPlayRequest
   :members:

.. autoclass:: BenchmarkResult
   :members:

.. autoclass:: BenchmarkRuntimeRequest
   :members:

.. autoclass:: BenchmarkStartupRequest
   :members:

.. autoclass:: BenchmarkTrainingRequest
   :members:

Micro-Benchmark Dispatch
------------------------

.. autoclass:: MicrobenchmarkCommand
   :members:

.. autoclass:: MicrobenchmarkFactory
   :members:

.. autofunction:: run_microbenchmark_cli

Latency Micro-Benchmarks
------------------------

.. autoclass:: LatencyBenchmarkRunner
   :members:

.. autoclass:: LatencySample
   :members:

.. autoclass:: LatencyStatistics
   :members:

.. autofunction:: add_latency_measurements

.. autofunction:: measure_latency

.. autofunction:: summarize_latency

Workflow Functions
------------------

.. autofunction:: run_benchmark

.. autofunction:: run_play_benchmark

.. autofunction:: run_runtime_benchmark

.. autofunction:: run_startup_benchmark

.. autofunction:: run_training_benchmark

Benchmark Framework
-------------------

.. autoclass:: BaseIsaacLabBenchmark
   :members:

.. autoclass:: BenchmarkMonitor
   :members:

.. autoclass:: MethodBenchmarkDefinition
   :members:

.. autoclass:: MethodBenchmarkRunner
   :members:

.. autoclass:: MethodBenchmarkRunnerConfig
   :members:

.. autoclass:: PlayBundle
   :members:

.. autoclass:: RuntimeBundle
   :members:

.. autoclass:: StartupBundle
   :members:

.. autoclass:: TrainingBundle
   :members:
