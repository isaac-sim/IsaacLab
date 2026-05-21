# envguard — LD_PRELOAD shim for thread-safe glibc env functions

## What it does

`libenvguard.so` is a tiny `LD_PRELOAD` shim that serializes the glibc
environment functions behind a single process-wide rwlock:

| Function        | Lock held |
| --------------- | --------- |
| `getenv`        | read      |
| `secure_getenv` | read      |
| `setenv`        | write     |
| `unsetenv`      | write     |
| `putenv`        | write     |
| `clearenv`      | write     |

The real symbols are resolved once via `dlsym(RTLD_NEXT, ...)` in a library
constructor; every wrapper just takes the lock and forwards to the real
implementation.

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
the environment. The crash reproduces only on busy CI shared runners
because that is where the thread schedule lines up badly.

We do not own libcarb, so we cannot fix the callers directly. Wrapping
glibc via `LD_PRELOAD` is the standard workaround and is what e.g. ASAN's
runtime does internally for the same reason.

## Building

```bash
cd docker/envguard
make
```

This produces `libenvguard.so` in the current directory using
`gcc -O2 -fPIC -shared -ldl -lpthread`. The shim has no dependencies
beyond `libc6-dev`, so it builds with whatever toolchain the Docker
base image already has.

To install it system-wide inside a container:

```bash
make install PREFIX=/usr/local   # -> /usr/local/lib/libenvguard.so
```

## Activation

There are two activation mechanisms; **use `/etc/ld.so.preload` in CI**.

### `/etc/ld.so.preload` (recommended in CI)

Write the absolute path of the shim into the system-wide preload file:

```bash
echo /usr/local/lib/libenvguard.so | sudo tee /etc/ld.so.preload
```

The dynamic linker reads this file at every process start and loads its
libraries unconditionally — **a child cannot turn it off by setting
`LD_PRELOAD` to something else**. This is what we need because Isaac Sim's
`_isaac_sim/python.sh` does exactly that, on its line 66:

```bash
# WAR for missing libcarb.so
export LD_PRELOAD=$SCRIPT_DIR/kit/libcarb.so
```

Any wrapper that relied on env-var-only `LD_PRELOAD` would be stripped at
that point, and the crash returns. We confirmed this empirically by
running `test_envguard` from a shell that overwrites `LD_PRELOAD` between
parent and child — the test SIGSEGVs again (exit 139), reproducing the
original CI signature.

Our `docker/Dockerfile.base` and
`source/isaaclab/test/install_ci/Dockerfile.installci` both write this file
at image-build time, so every test process inside the container
automatically inherits the shim regardless of what `LD_PRELOAD` is later
set to.

To remove the shim:

```bash
sudo rm /etc/ld.so.preload
```

### `LD_PRELOAD` (single-process activation, e.g. local debugging)

For a one-off invocation outside Docker — for example reproducing the bug
on a workstation — `LD_PRELOAD` is enough as long as the target process
does not exec a wrapper that overrides it:

```bash
LD_PRELOAD=/path/to/libenvguard.so my-test-command
```

Disable for a specific command:

```bash
LD_PRELOAD= some-command
```

## Caveats

* This only protects callers that go through libc's published symbols. Code
  that reads `__environ` (or `environ`) directly is **not** protected.
  In practice the crashes we see are inside `getenv`, so wrapping the
  public API is sufficient.
* `LD_PRELOAD` is ignored for setuid / setgid binaries; `/etc/ld.so.preload`
  is honored for both (subject to the linker's path restrictions, which
  `/usr/local/lib/libenvguard.so` satisfies). None of our test processes
  are setuid, so this is mostly academic.
* The shim is glibc-targeted but builds and runs unchanged on musl; the
  `secure_getenv` resolution is best-effort and falls back to `getenv`
  when the symbol is absent.

## Updating an existing container image

If the image was built before this change, you must rebuild it -
`/etc/ld.so.preload` is baked in at image-build time. On self-hosted CI
runners that cache previously-built images (see
`.github/actions/docker-build/action.yml`, which skips the build when a
matching tag exists locally), a fresh commit SHA forces a rebuild. You
can also force-rebuild manually with `docker build --no-cache`.
