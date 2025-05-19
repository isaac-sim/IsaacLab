# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


def log_info(text: str):
    print("[INFO] " + text)


def log_warn(text: str):
    print(f"\033[1;33m[WARNING] {text}\033[0m")


def log_error(text: str):
    print(f"\033[0;31m[ERROR] {text}\033[0m")
