# IsaacLab Tools

This directory contains utility scripts for working with IsaacLab simulations, assets, and configurations.

## 🛠️ Available Tools

### Robot Configuration Tools

#### `validate_robot_config.py`
Validate robot configuration files before running simulations.

```bash
# Basic validation
./isaaclab.sh -p scripts/tools/validate_robot_config.py --config isaaclab_assets.CRAZYFLIE_CFG

# Verbose output with detailed checks
./isaaclab.sh -p scripts/tools/validate_robot_config.py --config isaaclab_assets.ANYMAL_D_CFG --verbose
```

**Features:**
- Checks required and recommended attributes
- Validates data types and formats
- Verifies path structures and vector dimensions
- Provides clear error messages with emoji indicators

---

#### `export_robot_info.py`
Export robot configuration details to JSON format.

```bash
# Export to console
./isaaclab.sh -p scripts/tools/export_robot_info.py --config isaaclab_assets.CRAZYFLIE_CFG

# Export to file with pretty formatting
./isaaclab.sh -p scripts/tools/export_robot_info.py --config isaaclab_assets.FRANKA_PANDA_CFG --output franka.json --pretty
```

**Use Cases:**
- Documentation generation
- Configuration analysis
- Debugging robot setups
- Sharing robot specifications

---

### Performance Tools

#### `benchmark_performance.py`
Measure simulation performance across different robot counts.

```bash
# Default benchmark
./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot isaaclab_assets.CRAZYFLIE_CFG

# Custom robot counts
./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot isaaclab_assets.ANYMAL_D_CFG --counts 1,10,50,100,200 --iterations 1000

# Export results to CSV
./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot isaaclab_assets.CRAZYFLIE_CFG --output results.csv
```

**Metrics:**
- Total simulation time
- Average step time (ms)
- Frames per second (FPS)
- Steps per second throughput

**Features:**
- Warmup phase for accurate measurements
- CSV export for analysis
- GPU synchronization for precise timing
- Grid-based robot spawning

---

#### `check_instanceable.py`
Check if assets are properly instanced for optimal performance.

```bash
./isaaclab.sh -p scripts/tools/check_instanceable.py <path_to_usd> -n 4096 --physics
```

---

### Asset Conversion Tools

#### `convert_urdf.py`
Convert URDF files to USD format for use in Isaac Sim.

#### `convert_mjcf.py`
Convert MJCF (MuJoCo) files to USD format.

#### `convert_mesh.py`
Convert mesh files to optimized formats.

#### `convert_instanceable.py`
Convert USD assets to use instancing for better performance.

---

### Data Tools

#### `record_demos.py`
Record demonstration data with teleoperation.

#### `replay_demos.py`
Replay recorded demonstrations.

#### `merge_hdf5_datasets.py`
Merge multiple HDF5 datasets into one.

#### `hdf5_to_mp4.py`
Convert HDF5 data to MP4 videos.

#### `mp4_to_hdf5.py`
Convert MP4 videos to HDF5 format.

---

### Other Utilities

#### `blender_obj.py`
Work with Blender OBJ files.

#### `process_meshes_to_obj.py`
Process and convert meshes to OBJ format.

#### `pretrained_checkpoint.py`
Manage pretrained model checkpoints.

---

## 📊 Tool Comparison

| Tool | Purpose | Output | Simulation Required |
|------|---------|--------|--------------------- |
| `validate_robot_config.py` | Configuration validation | Console/Exit code | No |
| `export_robot_info.py` | Config export | JSON file | No |
| `benchmark_performance.py` | Performance testing | Console/CSV | Yes |
| `check_instanceable.py` | Asset optimization check | Console | Yes |

---

## 🚀 Quick Start Examples

### Before Running Simulations

1. **Validate your robot config:**
   ```bash
   ./isaaclab.sh -p scripts/tools/validate_robot_config.py --config your_robot.YOUR_CFG --verbose
   ```

2. **Export config for documentation:**
   ```bash
   ./isaaclab.sh -p scripts/tools/export_robot_info.py --config your_robot.YOUR_CFG --output docs/robot_specs.json --pretty
   ```

### Performance Optimization

3. **Benchmark your setup:**
   ```bash
   ./isaaclab.sh -p scripts/tools/benchmark_performance.py --robot your_robot.YOUR_CFG --counts 1,10,50,100 --output benchmark.csv
   ```

4. **Check asset instancing:**
   ```bash
   ./isaaclab.sh -p scripts/tools/check_instanceable.py path/to/asset.usd -n 1000 --physics
   ```

---

## 💡 Tips

- **Use `--verbose`** with validation tools for detailed information
- **Export benchmarks to CSV** for plotting and analysis in tools like Python/Excel
- **Run warmup iterations** before benchmarking for consistent results
- **Check instancing** for assets you plan to spawn in large quantities

---

## 🤝 Contributing

When adding new tools:
1. Follow the existing code structure and style
2. Include proper license headers (BSD-3-Clause)
3. Add comprehensive docstrings and usage examples
4. Update this README with your tool's documentation
5. Test with multiple robot configurations

---

## 📝 License

All tools follow the BSD-3-Clause license unless otherwise specified.
See individual files for details.
