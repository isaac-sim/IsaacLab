#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Print a short memory and storage summary of the CI host. Usage: runner_health.sh <label>

set -u

gib() { awk -v b="$1" 'BEGIN { printf "%.1f GiB", b / 1073741824 }'; }

read -r _ mem_total mem_used mem_free _ mem_cache mem_avail < <(free -b | awk '/^Mem:/')
read -r _ swap_total swap_used _ < <(free -b | awk '/^Swap:/')
docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)

echo "=== Runner health: ${1:-} ==="
echo "Memory  total $(gib "$mem_total") | used $(gib "$mem_used") | free $(gib "$mem_free") | cache $(gib "$mem_cache") | available $(gib "$mem_avail")"
echo "Swap    used $(gib "$swap_used") of $(gib "$swap_total")"
echo "Disk"
df -h --output=target,size,used,avail,pcent / "$docker_root" | awk '!seen[$1]++ { print "  " $0 }'
echo "Docker"
docker system df --format '{{.Type}}: {{.Size}}, reclaimable {{.Reclaimable}}' 2>/dev/null | sed 's/^/  /'
