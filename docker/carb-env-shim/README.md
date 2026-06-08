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

## Where the binary comes from

The shim ships inside Isaac Sim's Carbonite, so Isaac Lab no longer vendors a
prebuilt binary. `docker/Dockerfile.base` copies it out of the Isaac Sim
install tree at image-build time, looking for it under the Carbonite locations:

```text
${ISAACSIM_ROOT_PATH}/kit/kernel/plugins/libcarb.env.shim.so
${ISAACSIM_ROOT_PATH}/kit/libcarb.env.shim.so
${ISAACSIM_ROOT_PATH}/_build/linux-x86_64/release/kit/kernel/plugins/libcarb.env.shim.so
${ISAACSIM_ROOT_PATH}/_build/target-deps/carb_sdk_plugins/_build/linux-x86_64/release/libcarb.env.shim.so
```

The first match is used, with a recursive `find` under `${ISAACSIM_ROOT_PATH}`
as a fallback. If no shim is found, the build proceeds without registering
`/etc/ld.so.preload`. Because the source is the Carbonite that Isaac Sim was
built against, the shim automatically tracks the Isaac Sim version — there is
nothing to rebuild or refresh in this repo.

## Activation in Docker

`docker/Dockerfile.base` installs the shim to
`/usr/local/lib/libcarb.env.shim.so` and registers it via
`/etc/ld.so.preload` at image-build time.

`source/isaaclab/test/install_ci/Dockerfile.installci` does **not** register
the shim: it installs Isaac Sim per-test via pip into throwaway uv/conda
environments, so there is no Carbonite install tree to copy from at
image-build time. The shim ships bundled with the pip-installed `isaacsim`.

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

`/etc/ld.so.preload` is baked in at image-build time, copied from the Isaac
Sim Carbonite then in use. Rebuild the image after bumping the Isaac Sim base
image to pick up a newer shim. On self-hosted CI runners that cache
previously-built images, a fresh commit SHA forces a rebuild. You can also
force-rebuild manually with `docker build --no-cache`.
