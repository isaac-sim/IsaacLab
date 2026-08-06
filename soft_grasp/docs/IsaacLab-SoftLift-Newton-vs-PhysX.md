# Isaac Lab 软体抓取（Isaac-Lift-Soft-Franka-v0）：环境搭建 / 运行 / Newton vs PhysX 对比与性能基准

> 记录日期：2026-06-25
> 硬件：NVIDIA RTX 4090 (24 GB)，驱动 591.86 / CUDA 13.1，Windows 11
> 软件：Isaac Lab 3.0.0-beta2 + Isaac Sim 6.0.0.1，Python 3.12，PyTorch 2.11.0+cu128

---

## 0. TL;DR（结论先行）

- 环境用 **uv** 搭好了，`Isaac-Lift-Soft-Franka-v0` 示例**可以正常运行**（默认 Newton 后端，机械臂能稳定抓起并抬升软体）。
- 这个任务支持两个物理后端：**Newton（MJWarp 刚体 + VBD 软体，默认）** 和 **PhysX（FEM 软体）**。
- **在这台 Windows 机器上，PhysX 的吞吐和可扩展性远超 Newton**（峰值约 **65× 吞吐**、约 **12× 环境数**）。这与"Newton 是可扩展默认后端"的常规预期**相反**，根因是 **Newton 在本机 CUDA graph 捕获失败**（Windows 特有问题），退化成 eager 执行。
- **训练吞吐推荐：PhysX + 约 2048~3072 个环境（~6.6万~7.4万 env-steps/s）。**

---

## 1. 环境搭建（uv）

虚拟环境用 uv 创建，**放在短路径 `C:\iue`（不在仓库里、不在 OneDrive 里）**——这点很关键，见 [§5 已知问题](#5-已知问题--踩过的坑)。

```powershell
# 1) 装 uv
irm https://astral.sh/uv/install.ps1 | iex

# 2) 建 venv（Python 3.12）
uv venv --python 3.12 --seed C:\iue

# 3) 装 Isaac Sim 6.0.0.1（多 GB，含全部扩展缓存）
$env:VIRTUAL_ENV = "C:\iue"
uv pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

# 4) 装 CUDA 版 PyTorch（必须匹配 Isaac Sim 的硬性 pin：2.11.0）
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 5) 从源码装 Isaac Lab 各子模块（editable）
cd <repo>\IsaacLab-3.0.0-beta2
.\isaaclab.bat --install

# 6) 安装器会把 torch 强降到 2.10.0，装完后必须再装回 2.11.0（见 §5）
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

验证：`torch 2.11.0+cu128`、`cuda available: True`、`isaacsim 6.0.0.1`。

---

## 2. 运行示例

每次运行前设置环境变量（venv 在仓库外，所以要显式指定 `VIRTUAL_ENV`）：

```powershell
$env:Path = "C:\Users\jiuer\.local\bin;$env:Path"
$env:VIRTUAL_ENV = "C:\iue"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

### 2.1 跑官方状态机示例（默认 Newton 后端）

```powershell
.\isaaclab.bat -p scripts\environments\state_machine\lift_franka_soft.py --num_envs 1
```

验证结果：抓取-抬升流程完整跑通，软体从桌面 ~0.05 m 被抬到 ~0.34 m，无报错。

### 2.2 关于"窗口"——Isaac Lab 3.0 的变化

**3.0 里窗口由 `--viz` 控制，默认关闭**；旧的 `--headless` 已废弃，去掉它**不会**自动开窗口。

```powershell
# 轻量 Newton 查看器（推荐给 Newton 软体任务）
.\isaaclab.bat -p scripts\environments\state_machine\lift_franka_soft.py --num_envs 1 --viz newton
# 完整 Omniverse RTX 视口（更重，首次启动慢）
.\isaaclab.bat -p scripts\environments\state_machine\lift_franka_soft.py --num_envs 1 --viz kit
```

官方脚本是无限循环，用 **Ctrl+Break** 停止。

---

## 3. Newton vs PhysX 后端对比

任务默认是 Newton；PhysX 通过 preset 选择。官方 demo 的 `parse_env_cfg` 只会给默认 Newton，要切 PhysX 需在代码里 `resolve_presets(cfg, selected=["physx"])`（已封装进 `lift_franka_soft_compare.py`）。

| | **Newton**（默认） | **PhysX** |
|---|---|---|
| 求解器 | `CoupledMJWarpVBDSolverCfg`：MJWarp 刚体 + VBD 软体，双向耦合 | PhysX FEM 软体 |
| 材料写法 | Lamé 参数 `k_mu`/`k_lambda` | 杨氏模量 E + 泊松比 ν + 摩擦 |
| 物理参数 | 同一块立方体 0.3×0.05×0.05，E=8e4，ν=0.25，密度 300（两者等价）| 同上 |
| 多环境复制 | `replicate_physics=True` ✅ | `replicate_physics=False` ❌ |
| 单环境步进 | ~52 步/秒 | ~67~71 步/秒 |
| 状态机抓取结果 | **完整抓起并抬到 ~0.34 m**（状态到 5）| **没抓起来**，物体停在桌面 ~0.05 m（状态卡在 2~3）|

三个最值得注意的不同：

1. **同一套状态机，行为差很多。** 这个 demo 的开环状态机是**按 Newton 例子调的**，所以在 Newton 上稳定抓取+抬起；同样的脚本在 PhysX 上抓不住（接触响应/质心反馈不同，Newton 调好的阈值不能直接迁移）。PhysX **能跑**，但现成脚本在 PhysX 上不会成功抓取——这是后端特性差异 + 控制器没针对 PhysX 重调，不是 bug。
2. **多环境复制是 Newton 的设计卖点**（理论上跨环境批量求解可变形体）；PhysX 的可变形体**不能复制**，每个环境独立创建。**但在本机实测中（见 §4），这个理论优势被 Newton 的 Windows CUDA graph 问题完全抵消甚至反超。**
3. **Windows 上录视频的差异：** PhysX 走 Kit 的 RTX 相机，headless 直接能录；Newton 用自己的 OpenGL(pyglet) 查看器，**headless 模式依赖 EGL（只有 Linux 有）**，在 Windows 上报 `Library "EGL" not found`。workaround 是把 Newton 的 `ViewerGL` 强制成带窗口的 WGL 模式（见 `lift_franka_soft_compare.py`）。

录制的对比视频：
- `IsaacLab-3.0.0-beta2\videos\compare\newton\lift_soft_newton-step-0.mp4`
- `IsaacLab-3.0.0-beta2\videos\compare\physx\lift_soft_physx-step-0.mp4`

---

## 4. 性能基准（最大环境数 + 吞吐）

方法：headless、恒定动作（空载步进），每点单独启动一次 Isaac Sim。`env-steps/s = steps/s × num_envs`；显存为设备占用（total − free，含基线）。脚本：`scripts/environments/state_machine/bench_one.py`。

### 4.1 Newton（MJWarp + VBD）

| num_envs | steps/s | env-steps/s | 显存 |
|---:|---:|---:|---:|
| 1 | 51.8 | 52 | 1.8 GB |
| 16 | 42.7 | 683 | 1.9 GB |
| 64 | 17.8 | **1,141**（峰值）| 2.7 GB |
| 128 | 6.1 | 775 | 5.8 GB |
| 256 | 1.5 | 380 | 17.3 GB |
| 384 | **OOM**（还需 +7 GB，已用 23.8/24.6 GB）| | |
| 1024 | FAIL — Warp int32 数组上限 | | |

- **最大可用 ≈ 256 个环境**（显存受限），但吞吐在 ~64 个环境就见顶，之后断崖式下跌。
- 每步速率随环境数暴跌：52 → 18 → 6 → 1.5；显存超线性增长：2.7 → 5.8 → 17.3 GB。

### 4.2 PhysX（FEM 软体）

| num_envs | steps/s | env-steps/s | 显存 |
|---:|---:|---:|---:|
| 1 | 71.4 | 71 | 4.8 GB |
| 64 | 66.0 | 4,222 | 4.8 GB |
| 256 | 59.3 | 15,182 | 4.8 GB |
| 512 | 53.4 | 27,355 | 5.0 GB |
| 1024 | 43.7 | 44,765 | 5.1 GB |
| 2048 | 32.2 | 65,925 | 5.5 GB |
| 3072 | 24.1 | **74,139**（峰值）| 6.0 GB |
| 4096 | FAIL — 内部张量形状 bug（**不是 OOM**，仅用 2.5 GB）| | |

- **最大可用 ≈ 3072 个环境**，而且**不是显存受限**（3072 时才用 ~6 GB）。4096 的失败是软件 bug（`[4096,20,3]` vs `[4095,65,3]`），显存还很充足——bug 修了应该能开更多。
- 近似线性扩展，每步速率从 71 缓降到 24。

### 4.3 对比结论

| 指标 | Newton | PhysX | PhysX 优势 |
|---|---:|---:|---:|
| 峰值吞吐 (env-steps/s) | ~1,141 @64 | ~74,139 @3072 | **~65×** |
| 最大环境数 | ~256（显存）| ~3072（bug，非显存）| **~12×** |
| 峰值时显存 | 17.3 GB | 6.0 GB | 更省 |

**在这台 RTX 4090 / Windows 上，PhysX 用于该软体任务全面胜出**，与常规预期相反。

根因（基本是 Windows 特有问题）：**Newton 的 CUDA graph 捕获在本机失败**（日志 `libcudart not available` → 退化为 eager 执行），导致 Newton 跑得没有加速、显存暴涨（256 环境就 17 GB）、384 直接 OOM；此外还有 Warp 32 位数组上限（~512+）。在 **Linux** 且 CUDA graph 正常时，Newton 大概率会扩展得好得多——所以这是"**本机当前配置下**"的结论，不是 Newton vs PhysX 的普适排名。

---

## 5. 已知问题 / 踩过的坑

1. **Windows DLL 长路径崩溃（最关键）。** venv 一开始放在 OneDrive 深层路径里，Isaac Sim 的原生 `.pyd`/DLL 全部报 `The filename or extension is too long`，即使注册表 `LongPathsEnabled=1` 也没用（DLL 加载器不认）。**解决：把 venv 移到短路径 `C:\iue`。** 同时也避免了 OneDrive 去同步 10GB+ 的 venv。

2. **torch 版本冲突。** Isaac Sim 6.0.0.1 硬性要求 `torch==2.11.0`，但 `isaaclab.bat --install` 内部会强制把 torch 降到 2.10.0 并删掉 torchaudio。**装完后必须再装回 2.11.0 全套**（用 `--reinstall-package`，因为 uv 认为裸版本号 2.11.0 已满足、会跳过）。

3. **Newton 录视频在 Windows 上需要 EGL**（Linux 专属），headless 会失败。workaround：强制窗口化 WGL（见对比脚本）。

4. **Newton 在本机的 CUDA graph 失败**（`libcudart not available`），是其性能差的主因；属 Windows 侧问题，值得后续排查或改用 Linux。

5. **PhysX 在 4096 环境有内部 bug**（张量形状不匹配），并非显存瓶颈。

---

## 6. 相关文件清单

仓库：`IsaacLab-3.0.0-beta2/`

| 文件 | 说明 |
|---|---|
| `scripts/environments/state_machine/lift_franka_soft.py` | 官方示例（无限循环状态机）|
| `scripts/environments/state_machine/lift_franka_soft_verify.py` | 有界自检版（跑固定步数后退出，可删）|
| `scripts/environments/state_machine/lift_franka_soft_compare.py` | Newton/PhysX 对比 + 录像（`--backend`、`--video`）|
| `scripts/environments/state_machine/bench_one.py` | 单点性能基准（吞吐 + 显存 + OOM 处理）|
| `videos/compare/newton/lift_soft_newton-step-0.mp4` | Newton 抓取录像 |
| `videos/compare/physx/lift_soft_physx-step-0.mp4` | PhysX 录像 |

任务定义：`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_franka_soft/franka_soft_env_cfg.py`

---

## 7. 复现命令

```powershell
$env:Path = "C:\Users\jiuer\.local\bin;$env:Path"; $env:VIRTUAL_ENV = "C:\iue"; $env:OMNI_KIT_ACCEPT_EULA = "YES"

# 对比 + 录像
.\isaaclab.bat -p scripts\environments\state_machine\lift_franka_soft_compare.py --backend newton --video
.\isaaclab.bat -p scripts\environments\state_machine\lift_franka_soft_compare.py --backend physx  --video

# 单点基准（示例：PhysX 2048 环境）
.\isaaclab.bat -p scripts\environments\state_machine\bench_one.py --backend physx --num_envs 2048
```
