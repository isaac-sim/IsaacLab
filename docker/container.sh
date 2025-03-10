#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# print warning of deprecated script in yellow
echo -e "\e[33m------------------------------------------------------------"
echo -e "WARNING: This script is deprecated and will be removed in the future. Please use 'docker/container.py' instead."
echo -e "------------------------------------------------------------\e[0m\n"

# obtain current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# call the python script
python3 "${SCRIPT_DIR}/container.py" "${@:1}"
