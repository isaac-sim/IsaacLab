# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Build the libenvguard.so LD_PRELOAD shim.
#
# Usage:
#   make             # build libenvguard.so in this directory
#   make install     # install to $(PREFIX)/lib/libenvguard.so (PREFIX defaults to /usr/local)
#   make check       # build and run a concurrency smoke test under LD_PRELOAD
#   make clean       # remove build artifacts

CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -fPIC -fvisibility=default
LDFLAGS ?= -shared -Wl,--no-as-needed
LDLIBS  ?= -ldl -lpthread

PREFIX  ?= /usr/local
LIBDIR  ?= $(PREFIX)/lib

TARGET    = libenvguard.so
SOURCES   = envguard.c
TEST_BIN  = test_envguard
TEST_SRC  = test_envguard.c

.PHONY: all install check clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BIN): $(TEST_SRC)
	$(CC) -O2 -pthread -o $@ $^

# Run the smoke test under the shim. The plain (no-LD_PRELOAD) baseline
# is expected to segfault on glibc, so we don't fail the build on it.
check: $(TARGET) $(TEST_BIN)
	LD_PRELOAD=$(CURDIR)/$(TARGET) ./$(TEST_BIN)

install: $(TARGET)
	install -d $(DESTDIR)$(LIBDIR)
	install -m 0755 $(TARGET) $(DESTDIR)$(LIBDIR)/$(TARGET)

clean:
	rm -f $(TARGET) $(TEST_BIN)
