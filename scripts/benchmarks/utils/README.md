# Benchmark Utilities

This directory contains utility modules used by the benchmark scripts.

## Files

### `benchmark_utils.py`
Common utility functions for both kit and standalone modes. Provides factory functions that return the appropriate logging implementations based on the mode:
- `create_kit_logging_functions()`: Returns logging functions for Isaac Sim kit mode
- `create_standalone_logging_functions()`: Returns logging functions for standalone mode
- `get_timer_value()`: Safely retrieves timer values

### `standalone_benchmark.py`
Standalone benchmark infrastructure that replicates the functionality of `isaacsim.benchmark.services` without requiring Isaac Sim. Contains:
- Measurement classes (`SingleMeasurement`, `ListMeasurement`, `DictMeasurement`, `BooleanMeasurement`)
- Metadata classes (`StringMetadata`, `IntMetadata`, `FloatMetadata`, `DictMetadata`)
- `TestPhase` class for organizing measurements
- Backend implementations:
  - `OmniPerfKPIFile`: Single JSON with all phases
  - `OsmoKPIFile`: Separate JSON per phase
  - `JSONFileMetrics`: Detailed JSON with full objects
  - `LocalLogMetrics`: Console output only
- `StandaloneBenchmark`: Main benchmark class
- **System Metrics Collection** (requires `psutil` and optionally `GPUtil`/`pynvml`):
  - CPU metrics: user, system, idle, iowait percentages
  - Memory metrics: RSS, VMS, USS (in GB)
  - GPU metrics: memory usage, utilization
  - Runtime duration per phase
  - System information: CPU count, GPU device name

### `utils.py`
Legacy utility functions for Isaac Sim kit mode benchmarking. Contains:
- Version retrieval functions (`get_isaaclab_version`, `get_newton_version`, `get_mujoco_warp_version`)
- Logging functions for various metrics (timing, rewards, episode lengths, etc.)
- `parse_tf_logs()`: TensorBoard log parser

## Usage

The benchmark scripts (`benchmark_non_rl.py`, `benchmark_rlgames.py`, `benchmark_rsl_rl.py`) automatically select the appropriate utilities based on the `--kit` flag:

```python
if args_cli.kit:
    # Use Isaac Sim benchmark services
    log_funcs = create_kit_logging_functions()
else:
    # Use standalone benchmark services
    log_funcs = create_standalone_logging_functions()
```

This approach eliminates code duplication across the benchmark scripts while maintaining support for both modes.

## System Metrics Collection

The standalone benchmark can automatically collect system metrics similar to Isaac Sim's benchmark services:

### Collected Metrics

**System Information (collected once):**
- Number of CPUs
- GPU device name

**Runtime Metrics (collected per phase):**
- **Memory:**
  - RSS (Resident Set Size)
  - VMS (Virtual Memory Size)
  - USS (Unique Set Size)
- **CPU Usage:**
  - User time percentage
  - System time percentage
  - Idle time percentage
  - I/O wait time percentage
- **GPU (if available):**
  - Memory used
  - Total memory
  - GPU utilization percentage
- **Runtime:** Phase execution duration

### Requirements

- **psutil**: Required for CPU and memory metrics (automatically detected)
- **GPUtil** or **pynvml**: Optional for GPU metrics (automatically detected)
- **nvidia-smi**: Fallback for GPU metrics (usually pre-installed with NVIDIA drivers)

The GPU detection tries three methods in order:
1. **GPUtil**: Fast Python library
2. **pynvml** (nvidia-ml-py3): Official NVIDIA library
3. **nvidia-smi**: Direct system call (most reliable, works even with driver/library version mismatch)

Install with:
```bash
pip install psutil GPUtil
# or
pip install psutil nvidia-ml-py3
```

Note: `nvidia-smi` is automatically used as a fallback and doesn't require installation (comes with NVIDIA drivers).

### Usage

System metrics are collected automatically by default:

```python
benchmark = StandaloneBenchmark(
    benchmark_name="my_benchmark",
    collect_system_metrics=True  # default
)
```

To disable system metrics:
```python
benchmark = StandaloneBenchmark(
    benchmark_name="my_benchmark",
    collect_system_metrics=False
)
```

The metrics will appear in your output JSON alongside your custom measurements.
