// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// envguard: preload shim that makes glibc's environment functions
// thread-safe relative to each other.
//
// POSIX explicitly does NOT guarantee that getenv() is safe to call
// concurrently with setenv(), putenv(), unsetenv(), or clearenv() on
// another thread. glibc walks the `__environ` array in-place inside
// getenv(), so a concurrent setenv() that reallocs the array produces
// SIGSEGV with a stack like:
//
//     libc.so.6!__sigaction+0x50
//     libc.so.6!getenv+0x56
//     <some Carb / Kit / Omni plugin>
//
// Isaac Lab CI hits this race intermittently because Carb / Kit / Omni
// plugins call getenv() from worker threads while the main thread (or
// Python's site initialization, or another plugin) is still mutating
// the environment.
//
// This shim resolves the real glibc symbols via dlsym(RTLD_NEXT, ...)
// and serializes them with a single process-wide rwlock:
//   * readers: getenv, secure_getenv
//   * writers: setenv, unsetenv, putenv, clearenv
//
// Activation:
//   * In CI Docker images: written to /etc/ld.so.preload at image-build
//     time. The dynamic linker honors that file unconditionally for every
//     process, which is required because Isaac Sim's _isaac_sim/python.sh
//     overwrites LD_PRELOAD with kit/libcarb.so before exec'ing python.
//   * For local one-off use: LD_PRELOAD=/path/to/libenvguard.so cmd.
//
// The shim is intentionally tiny and has zero deps beyond libc + pthread
// + libdl so it can be built early in a Docker stage with nothing more
// than `gcc` and `libc6-dev`.

#define _GNU_SOURCE

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Single rwlock guarding the whole __environ table. A rwlock (rather than a
// plain mutex) keeps the common case -- many threads concurrently calling
// getenv() with no writers -- contention-free.
static pthread_rwlock_t g_env_lock = PTHREAD_RWLOCK_INITIALIZER;

typedef char *(*getenv_fn_t)(const char *);
typedef char *(*secure_getenv_fn_t)(const char *);
typedef int (*setenv_fn_t)(const char *, const char *, int);
typedef int (*unsetenv_fn_t)(const char *);
typedef int (*putenv_fn_t)(char *);
typedef int (*clearenv_fn_t)(void);

static getenv_fn_t real_getenv;
static secure_getenv_fn_t real_secure_getenv;
static setenv_fn_t real_setenv;
static unsetenv_fn_t real_unsetenv;
static putenv_fn_t real_putenv;
static clearenv_fn_t real_clearenv;

// Resolve a symbol via RTLD_NEXT, aborting if it cannot be found. We only
// abort for genuinely required symbols (everything except secure_getenv,
// which is glibc-specific and may be absent on non-glibc systems).
static void *resolve_required(const char *name)
{
    void *sym = dlsym(RTLD_NEXT, name);
    if (sym == NULL) {
        fprintf(stderr, "envguard: dlsym(RTLD_NEXT, \"%s\") failed: %s\n", name, dlerror());
        abort();
    }
    return sym;
}

// Resolve all real symbols once at library load time. Constructors registered
// with __attribute__((constructor)) run after libc has finished its own init,
// so dlsym is safe here.
__attribute__((constructor)) static void envguard_init(void)
{
    real_getenv = (getenv_fn_t) resolve_required("getenv");
    real_setenv = (setenv_fn_t) resolve_required("setenv");
    real_unsetenv = (unsetenv_fn_t) resolve_required("unsetenv");
    real_putenv = (putenv_fn_t) resolve_required("putenv");
    real_clearenv = (clearenv_fn_t) resolve_required("clearenv");
    // secure_getenv is a glibc extension; tolerate its absence.
    real_secure_getenv = (secure_getenv_fn_t) dlsym(RTLD_NEXT, "secure_getenv");
}

// Defensive lazy-init fallback: if a caller hits one of our wrappers before
// our constructor has finished (unlikely in practice, but possible with very
// early dlopen() chains), do the lookup on demand.
static inline void *resolve_lazy(void **slot, const char *name)
{
    void *sym = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (sym == NULL) {
        sym = dlsym(RTLD_NEXT, name);
        if (sym == NULL) {
            return NULL;
        }
        __atomic_store_n(slot, sym, __ATOMIC_RELEASE);
    }
    return sym;
}

char *getenv(const char *name)
{
    getenv_fn_t fn = (getenv_fn_t) resolve_lazy((void **) &real_getenv, "getenv");
    if (fn == NULL) {
        return NULL;
    }
    pthread_rwlock_rdlock(&g_env_lock);
    char *value = fn(name);
    pthread_rwlock_unlock(&g_env_lock);
    return value;
}

char *secure_getenv(const char *name)
{
    secure_getenv_fn_t fn = (secure_getenv_fn_t) resolve_lazy((void **) &real_secure_getenv, "secure_getenv");
    if (fn == NULL) {
        // Fall back to plain getenv if libc lacks secure_getenv.
        return getenv(name);
    }
    pthread_rwlock_rdlock(&g_env_lock);
    char *value = fn(name);
    pthread_rwlock_unlock(&g_env_lock);
    return value;
}

int setenv(const char *name, const char *value, int overwrite)
{
    setenv_fn_t fn = (setenv_fn_t) resolve_lazy((void **) &real_setenv, "setenv");
    if (fn == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&g_env_lock);
    int rc = fn(name, value, overwrite);
    pthread_rwlock_unlock(&g_env_lock);
    return rc;
}

int unsetenv(const char *name)
{
    unsetenv_fn_t fn = (unsetenv_fn_t) resolve_lazy((void **) &real_unsetenv, "unsetenv");
    if (fn == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&g_env_lock);
    int rc = fn(name);
    pthread_rwlock_unlock(&g_env_lock);
    return rc;
}

int putenv(char *string)
{
    putenv_fn_t fn = (putenv_fn_t) resolve_lazy((void **) &real_putenv, "putenv");
    if (fn == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&g_env_lock);
    int rc = fn(string);
    pthread_rwlock_unlock(&g_env_lock);
    return rc;
}

int clearenv(void)
{
    clearenv_fn_t fn = (clearenv_fn_t) resolve_lazy((void **) &real_clearenv, "clearenv");
    if (fn == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&g_env_lock);
    int rc = fn();
    pthread_rwlock_unlock(&g_env_lock);
    return rc;
}
