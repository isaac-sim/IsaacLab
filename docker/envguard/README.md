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

Set `LD_PRELOAD` to the absolute path of the shim before invoking any
process whose env-function callers are not thread-safe:

```bash
export LD_PRELOAD=/usr/local/lib/libenvguard.so
```

In our CI Docker images we set this as a container-level `ENV` so it is
inherited by every test process automatically. See
`docker/Dockerfile.base` and `source/isaaclab/test/install_ci/Dockerfile.installci`
for the integration points.

If you need to disable the shim for a specific process (e.g. for debugging)
just unset `LD_PRELOAD` for that command:

```bash
LD_PRELOAD= some-command
```

## Caveats

* This only protects callers that go through libc's published symbols. Code
  that reads `__environ` (or `environ`) directly is **not** protected.
  In practice the crashes we see are inside `getenv`, so wrapping the
  public API is sufficient.
* `LD_PRELOAD` is ignored for setuid / setgid binaries. None of our test
  processes are setuid, so this does not matter for us.
* The shim is glibc-targeted but builds and runs unchanged on musl; the
  `secure_getenv` resolution is best-effort and falls back to `getenv`
  when the symbol is absent.
