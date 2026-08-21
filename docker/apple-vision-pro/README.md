# Run Isaac Lab on Apple Vision Pro with Docker

This runbook starts Isaac Sim and Isaac Lab in one headless Docker container, streams the stereo view through
CloudXR, and deploys the native Isaac XR Teleop Sample Client to Apple Vision Pro. The setup uses optical hand
tracking with the `IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs` task.

For the Snap Circuits workbench populated from private S3 and the two local
asset bundles, with selectable GR1T2, Sharpa Wave, and ProHand backends, follow
the dedicated [Snap Circuits demo](snap-circuits/README.md) after completing
the base setup below.

The configuration was validated on the following server:

| Component | Validated version |
| --- | --- |
| Host | Ubuntu 24.04 LTS, Linux x86_64 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| NVIDIA driver | 595.84 |
| Docker / Compose | 29.1.3 / 2.40.3 |
| Isaac Sim container | 6.0.1 |
| Isaac Teleop | 1.4.x |
| Client | Isaac XR Teleop Sample Client `v3.0.1` |

Isaac Lab's documented minimums still apply: Ubuntu 22.04 or newer, a recent NVIDIA production driver, 32 GB
RAM, and 16 GB GPU VRAM. XR teleoperation is supported on Linux x86_64. The server and headset must be directly
IP-reachable; a dedicated 5 GHz, 6 GHz, or Wi-Fi 6 access point is strongly recommended.

## 1. Check out this PR

Install Git, Docker Engine, Docker Compose, and the NVIDIA Container Toolkit on the Linux server. Confirm that
the host driver and Docker GPU runtime work before continuing:

```bash
nvidia-smi
docker version
docker compose version
```

Clone Isaac Lab, then check out the branch associated with this PR. From an existing upstream checkout, the
generic GitHub PR checkout form is:

```bash
git fetch origin pull/<PR_NUMBER>/head:avp-cloudxr-docker-setup
git switch avp-cloudxr-docker-setup
```

If pulling `nvcr.io/nvidia/isaac-sim:6.0.1` returns an authorization error, authenticate to NGC first:

```bash
docker login nvcr.io
```

Use `$oauthtoken` as the username and an NGC API key as the password.

## 2. Prepare the Linux host

CloudXR is latency-sensitive. Set the CPU governor to `performance` and verify it:

```bash
sudo apt-get update
sudo apt-get install -y linux-tools-common "linux-tools-$(uname -r)"
sudo cpupower frequency-set -g performance
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

The expected output is `performance`. Repeat this after a reboot unless the host makes the setting persistent.

Allow the Apple native CloudXR ports through the host firewall:

```bash
sudo ufw allow 48010/tcp
sudo ufw allow 48322/tcp
sudo ufw allow 47998/udp
sudo ufw allow 48005/udp
sudo ufw allow 48008/udp
sudo ufw allow 48012/udp
sudo ufw allow 47999/udp
sudo ufw allow 48000/udp
sudo ufw allow 48002/udp
# Optional Snap Circuits spectator camera for VLC/ffplay on the Mac:
sudo ufw allow 8554/tcp
sudo ufw status
```

The container uses host networking, so these rules belong on the host rather than inside the container.

Disable X11 forwarding for the headless SSH session:

```bash
printf '[X11]\nx11_forwarding_enabled = 0\n' > docker/.container.cfg
```

This avoids the `KeyError: 'DISPLAY'` failure produced when an old container configuration enables X11 but the
SSH session has no `DISPLAY` variable. XQuartz is not required for this headless workflow.

## 3. Build and start Isaac Lab

Review and accept the NVIDIA CloudXR SDK EULA once, then start the server from
the repository root:

```bash
./docker/apple-vision-pro/avp-teleop.sh accept-eula
./docker/apple-vision-pro/avp-teleop.sh start
```

The acceptance command links to the EULA and requires an explicit `YES`; its
marker and CloudXR logs live in the persistent `isaac-cloudxr` Docker volume.
The first run builds `isaac-lab-base` from the Isaac Sim 6.0.1 image. The image includes `libvulkan1` and
`libbsd0`, which the CloudXR runtime needs, and the normal `isaaclab.sh --install` step installs the `teleop`
extra (`isaacteleop`, CloudXR support, and `dex-retargeting`). Do not start the legacy
`docker-compose.cloudxr-runtime.patch.yaml` service; Isaac Lab 3 runs the runtime and simulation together in one
container.

The helper uses Docker directly when the current account can reach the daemon and falls back to `sudo` otherwise.
It starts the teleoperation process in the background. Check readiness with:

```bash
./docker/apple-vision-pro/avp-teleop.sh status
```

Wait until the output contains both:

```text
teleop=running
cloudxr_signaling=listening tcp/48010
```

The inherited Isaac Sim image health probe may label a headless container `unhealthy` even while the teleoperation
process and CloudXR signaling socket are ready. The two checks above are the readiness criteria for this workflow.

Optional package and version checks:

```bash
docker exec isaac-lab-base bash -lc 'echo "$ISAACSIM_VERSION"'
docker exec isaac-lab-base /isaac-sim/python.sh -m pip show isaacteleop dex-retargeting
docker exec isaac-lab-base dpkg-query -W libvulkan1 libbsd0
```

If Docker requires elevated access, prefix these three diagnostic commands with `sudo`.

## 4. Verify connectivity from the Mac

Find the Linux server's LAN address:

```bash
hostname -I
```

From the Mac that will deploy the Vision Pro client, verify the standard CloudXR signaling port:

```bash
nc -vz <ISAAC_LAB_SERVER_IP> 48010
```

Do not proceed until this succeeds. If it fails, check routing between the Wi-Fi network and server, `ufw status`,
and the helper's `status` and `logs` commands.

## 5. Build and install the Vision Pro client

The client requires:

- Apple Vision Pro running visionOS 26 or newer.
- An Apple Silicon Mac running macOS Sequoia 15.6 or newer.
- Xcode 26 or newer with the visionOS platform installed.
- A paid Apple Developer Program team. A free Personal Team cannot provision the Low-Latency Streaming entitlement.
- Acceptance of the latest Apple Developer Program License Agreement at
  <https://developer.apple.com/account>.

On the Mac, install Git LFS and clone the matching client release:

```bash
brew install git-lfs
git lfs install
git clone https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple.git
cd isaac-xr-teleop-sample-client-apple
git checkout v3.0.1
open IsaacXRTeleopClient.xcodeproj
```

Pair and deploy the device:

1. On Vision Pro, enable **Settings > Privacy & Security > Developer Mode**, then restart when prompted.
2. In Xcode, open **Window > Devices and Simulators**, select the Vision Pro, and finish pairing.
3. Select the top-level `IsaacXRTeleopClient` project and open **Signing & Capabilities**.
4. Leave **Automatically manage signing** enabled, select the paid developer team, and change the bundle identifier
   to a unique value such as `com.example.IsaacXRTeleopClient`.
5. Confirm that **Hands Tracking**, **Low-Latency Streaming**, and **Microphone** remain present. Removing
   Low-Latency Streaming may allow a free-team build, but it removes the capability this workflow needs.
6. Select **Product > Destination > Apple Vision Pro**, then choose **Product > Run**.
7. If visionOS reports an untrusted developer, open **Settings > General > Device Management**, trust the developer
   certificate, and relaunch the app.
8. Approve the local-network and hand-tracking permission prompts.

If Xcode reports `PLA Update available`, accept the agreement in the developer account and retry. If it reports
that the Personal Team does not support Low-Latency Streaming, select the paid team rather than the Personal Team.

## 6. Connect and teleoperate

On Vision Pro:

1. Open **Isaac XR Teleop Client**.
2. Enter `<ISAAC_LAB_SERVER_IP>` without a protocol or port.
3. Select **Connect** and wait for the streamed simulation.
4. Select **Play**.
5. Move both hands to control the GR1T2 robot. Use **Stop** and **Reset** from the client as needed.

## Operations and troubleshooting

Use the helper for routine operation:

```bash
./docker/apple-vision-pro/avp-teleop.sh status
./docker/apple-vision-pro/avp-teleop.sh logs
./docker/apple-vision-pro/avp-teleop.sh stop
./docker/apple-vision-pro/avp-teleop.sh start
```

Detached startup output is retained at `docker/apple-vision-pro/teleop.log`
and included by the `logs` command.

`stop` ends only the teleoperation process. To stop the container while preserving its named cache volumes, run
`docker stop isaac-lab-base` (or `sudo docker stop isaac-lab-base` when required). The generic
`docker/container.py stop` command removes the named volumes and is therefore not appropriate when the caches or
recorded data must be retained.

Common failures:

| Symptom | Resolution |
| --- | --- |
| `KeyError: 'DISPLAY'` during container startup | Recreate `docker/.container.cfg` with X11 disabled as shown above. |
| Port 48010 is not listening | Wait for startup, then inspect `avp-teleop.sh logs`. Confirm `isaacteleop` is installed. |
| Mac cannot reach port 48010 | Check the server IP, host firewall, VLAN/client-isolation rules, and that both networks are routable. |
| Vision Pro connects but streaming is unstable | Use 5/6 GHz Wi-Fi with one wireless hop, restore the `performance` governor, and minimize packet loss. |
| Xcode cannot create the provisioning profile | Select a paid team, use a unique bundle identifier, and accept the latest program agreement. |
| Client has no hand tracking | Deploy to physical Vision Pro rather than the simulator and approve the Hands Tracking permission. |

For more detail, see the main
[CloudXR teleoperation guide](../../docs/source/how-to/cloudxr_teleoperation.rst), the
[Isaac XR Teleop Sample Client](https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple), and the
[CloudXR network requirements](https://docs.nvidia.com/cloudxr-sdk/release/6/requirement/network_setup.html).
