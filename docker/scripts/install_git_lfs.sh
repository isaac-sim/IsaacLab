#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Install the same upstream executable on each supported container architecture.
set -euo pipefail

version=3.8.0
architecture="$(dpkg --print-architecture)"
# https://github.com/git-lfs/git-lfs/releases/download/v3.8.0/sha256sums.asc
case "${architecture}" in
    amd64) checksum=e455e00f15d9b95661b8d53498ffb0c3367962cf1ec73c31ab7369516cd6ab8d ;;
    arm64) checksum=ac9c8efac980bb0505ead384d087e2acb6486fd8498691a2165fa174ec6118c2 ;;
    *) echo "Unsupported Git LFS architecture: ${architecture}" >&2; exit 1 ;;
esac

download_dir="$(mktemp -d)"
trap 'rm -rf -- "${download_dir}"' EXIT
archive="git-lfs-linux-${architecture}-v${version}.tar.gz"
wget --https-only --tries=3 --timeout=60 -O "${download_dir}/${archive}" \
    "https://github.com/git-lfs/git-lfs/releases/download/v${version}/${archive}"
printf '%s  %s\n' "${checksum}" "${download_dir}/${archive}" | sha256sum --check --strict -
tar -xzf "${download_dir}/${archive}" --no-same-owner -C "${download_dir}"
release_dir="${download_dir}/git-lfs-${version}"
"${release_dir}/git-lfs" version | grep -F "git-lfs/${version} ("

# Preserve upstream and vendored-component notices when replacing the distro package.
vendor_archive="git-lfs-vendor-v${version}.tar.gz"
wget --https-only --tries=3 --timeout=60 -O "${download_dir}/${vendor_archive}" \
    "https://github.com/git-lfs/git-lfs/releases/download/v${version}/${vendor_archive}"
printf '%s  %s\n' 28c49d50bea97d0b860fd1599cc5ab45aac937fd31125db8374bf149bb95622f \
    "${download_dir}/${vendor_archive}" | sha256sum --check --strict -
mkdir "${download_dir}/notices"
tar -xzf "${download_dir}/${vendor_archive}" --no-same-owner -C "${download_dir}/notices"
notice_root="${download_dir}/notices/git-lfs-${version}"
test -f "${notice_root}/LICENSE.md"

# Remove an inherited distro build as well: leaving it at /usr/bin would ship
# a second executable with its own older embedded Go runtime.
if dpkg-query -W -f='${Status}' git-lfs 2>/dev/null | grep -qx 'install ok installed'; then
    apt-get remove -y git-lfs
fi
install -m 0755 "${release_dir}/git-lfs" /usr/local/bin/git-lfs
# Configure filters for the non-root runtime user without changing any repository hooks.
"${release_dir}/git-lfs" install --system --skip-repo
install -D -m 0644 "${release_dir}/README.md" /usr/local/share/doc/git-lfs/README.md
while IFS= read -r -d '' manual; do
    install -D -m 0644 "${manual}" "/usr/local/share/man/${manual#"${release_dir}/man/"}"
done < <(find "${release_dir}/man" -type f \( -name '*.1' -o -name '*.7' \) -print0)
while IFS= read -r -d '' notice; do
    install -D -m 0644 "${notice}" "/usr/local/share/doc/git-lfs/${notice#"${notice_root}/"}"
done < <(find "${notice_root}" -type f \( -iname 'license*' -o -iname 'notice*' -o -iname 'copying*' \) -print0)
