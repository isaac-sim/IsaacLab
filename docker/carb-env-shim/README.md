# libcarb.env.shim — Carbonite preload shim for thread-safe glibc env functions

## What it does

`libcarb.env.shim.so` is a Carbonite-provided preload shim that serializes
the glibc environment functions behind a process-wide lock:

| Function        | Lock held |
| --------------- | --------- |
| `getenv`        | read      |
| `secure_getenv` | read      |
| `setenv`        | write     |
| `unsetenv`      | write     |
| `putenv`        | write     |
| `clearenv`      | write     |

## Why we need it

POSIX explicitly does **not** require `getenv()` to be thread-safe relative
to `setenv()` / `putenv()` / `unsetenv()` / `clearenv()`. glibc walks the
`__environ` array in-place inside `getenv`, so a concurrent mutation on
another thread can produce a SIGSEGV like the one we see intermittently in
Isaac Lab CI:

```text
[Fatal] [carb.crashreporter-breakpad.plugin] 000: libc.so.6!__sigaction+0x50 ***
[Fatal] [carb.crashreporter-breakpad.plugin] 001: libc.so.6!getenv+0x56 ***
[Fatal] [carb.crashreporter-breakpad.plugin] 002: libcarb.crashreporter-breakpad.plugin.so!...
...
Process killed by signal 11 (SIGSEGV — segmentation fault)
```

Carb / Kit / Omni plugins call `getenv()` from worker threads while the
main thread (or Python's startup, or another plugin) is still mutating
the environment.

## Prebuilt binaries

Isaac Lab ships prebuilt shims under:

```text
docker/carb-env-shim/linux-x86_64/libcarb.env.shim.so
docker/carb-env-shim/linux-aarch64/libcarb.env.shim.so   # add when available
```

The x86_64 binary is built from Carbonite (Kit) and exports the env
wrappers listed above. Rebuild and refresh the committed copy when bumping
the Carbonite version used by Isaac Sim.

To rebuild locally from a Kit tree:

```bash
# After a Kit release build:
cp kit/_build/linux-x86_64/release/kernel/plugins/libcarb.env.shim.so \
   docker/carb-env-shim/linux-x86_64/libcarb.env.shim.so
```

## Activation in Docker

Our `docker/Dockerfile.base` and
`source/isaaclab/test/install_ci/Dockerfile.installci` install the shim to
`/usr/local/lib/libcarb.env.shim.so` and register it via
`/etc/ld.so.preload` at image-build time.

We use `/etc/ld.so.preload` instead of `ENV LD_PRELOAD` because Isaac Sim's
`_isaac_sim/python.sh` unconditionally overwrites `LD_PRELOAD` with
`kit/libcarb.so` before exec'ing the interpreter, which would strip an
env-var-only preload. `/etc/ld.so.preload` is read by the dynamic linker
for every process and cannot be turned off by env-var manipulation.

To remove the shim from a running container:

```bash
sudo rm /etc/ld.so.preload
```

## Local debugging

For a one-off invocation outside Docker, preload the shim explicitly:

```bash
LD_PRELOAD=/path/to/libcarb.env.shim.so my-test-command
```

Disable for a specific command:

```bash
LD_PRELOAD= some-command
```

## Updating an existing container image

`/etc/ld.so.preload` is baked in at image-build time. Rebuild the image
after updating the committed binary. On self-hosted CI runners that cache
previously-built images, a fresh commit SHA forces a rebuild. You can also
force-rebuild manually with `docker build --no-cache`.
