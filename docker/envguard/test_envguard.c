// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Concurrency smoke test for libenvguard.so.
//
// Without LD_PRELOAD=libenvguard.so this program is expected to either
// crash or report inconsistent reads under the thread sanitizer. With
// the shim loaded it should run to completion and print a non-zero
// number of successful reads.
//
// Build: cc -O2 -pthread test_envguard.c -o test_envguard
// Run  : LD_PRELOAD=$PWD/libenvguard.so ./test_envguard

#define _GNU_SOURCE

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const int kIterations = 200000;
static const int kReaderThreads = 8;
static const int kWriterThreads = 4;
static const int kKeys = 32;

static atomic_int g_reads_ok = 0;
static atomic_int g_writes_ok = 0;
static atomic_int g_stop = 0;

static void *reader_thread(void *arg)
{
    (void) arg;
    char name[32];
    for (int i = 0; i < kIterations && !atomic_load(&g_stop); ++i) {
        snprintf(name, sizeof(name), "ENVGUARD_TEST_%d", i % kKeys);
        const char *v = getenv(name);
        if (v != NULL) {
            // Touch the value to force the libc strcmp / scan path to complete.
            volatile size_t l = strlen(v);
            (void) l;
            atomic_fetch_add(&g_reads_ok, 1);
        }
    }
    return NULL;
}

static void *writer_thread(void *arg)
{
    int tid = (int) (long) arg;
    char name[32];
    char value[32];
    for (int i = 0; i < kIterations && !atomic_load(&g_stop); ++i) {
        snprintf(name, sizeof(name), "ENVGUARD_TEST_%d", i % kKeys);
        snprintf(value, sizeof(value), "v%d_%d", tid, i);
        if (i % 4 == 0) {
            unsetenv(name);
        } else {
            setenv(name, value, 1);
        }
        atomic_fetch_add(&g_writes_ok, 1);
    }
    return NULL;
}

int main(void)
{
    pthread_t readers[kReaderThreads];
    pthread_t writers[kWriterThreads];

    for (int i = 0; i < kReaderThreads; ++i) {
        if (pthread_create(&readers[i], NULL, reader_thread, NULL) != 0) {
            fprintf(stderr, "pthread_create(reader) failed\n");
            return 1;
        }
    }
    for (int i = 0; i < kWriterThreads; ++i) {
        if (pthread_create(&writers[i], NULL, writer_thread, (void *) (long) i) != 0) {
            fprintf(stderr, "pthread_create(writer) failed\n");
            return 1;
        }
    }

    for (int i = 0; i < kReaderThreads; ++i) {
        pthread_join(readers[i], NULL);
    }
    for (int i = 0; i < kWriterThreads; ++i) {
        pthread_join(writers[i], NULL);
    }

    printf("envguard smoke test: reads_ok=%d writes_ok=%d\n", atomic_load(&g_reads_ok), atomic_load(&g_writes_ok));
    return 0;
}
