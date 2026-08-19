#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Body of the 'Run Tests in Docker Container' step. Kept in a file because a
# composite action's run: block is capped at 21000 characters (counted as
# chars + lines, i.e. CRLF), and this function alone far exceeds that.
# Positional arguments are passed through from action.yml.

# Function to run tests in Docker container
run_tests() {
  local test_path="$1"
  local result_file="$2"
  local container_name="$3"
  local image_tag="$4"
  local reports_dir="$5"
  local pytest_options="$6"
  local filter_pattern="$7"
  local exclude_pattern="$8"
  local curobo_only="$9"
  local include_files="${10}"
  local quarantined_only="${11}"
  local shard_index="${12}"
  local shard_count="${13}"
  local volume_mount_source="${14}"
  local extra_pip_packages="${15}"
  local test_node_ids_file="${16}"
  local test_node_ids_key="${17}"
  local wheelhouse_host_dir="${18}"
  local wheelhouse_packages="${19}"
  local test_k_expr="${20}"
  local ci_marker="${21}"
  local standalone_script_scope="${22}"
  local standalone_script_visualizer="${23}"
  local standalone_script_runtime_group="${24}"
  local warp_cache_host_dir="${25}"
  local extra_uv_packages="${26}"
  # The action-to-script positional interface is append-only for direct callers.
  local allow_online_fallback="${27:-true}"
  local logs_pid=""
  local wait_pid=""
  local docker_wait_file="/tmp/.docker_exit_${container_name}"
  local docker_runtime_dir=""
  local wheelhouse_error=""
  local -a wheelhouse_volume_args=()

  # Kill the container immediately if the runner is cancelled.
  # The GitHub Actions runner can deliver HUP, INT, or TERM on cancellation
  # (see actions/runner#1309). Trap all three so the cleanup always fires.
  # shellcheck disable=SC2064
  trap "echo '::warning::Cancellation signal - killing container ${container_name}'; \
        docker kill '${container_name}' 2>/dev/null || true; \
        docker rm -f '${container_name}' 2>/dev/null || true; \
        rm -f '${docker_wait_file}'; \
        if [ -n \"\$docker_runtime_dir\" ]; then rm -rf \"\$docker_runtime_dir\" 2>/dev/null || true; fi; \
        if [ -n \"\$logs_pid\" ]; then kill \"\$logs_pid\" 2>/dev/null || true; fi; \
        if [ -n \"\$wait_pid\" ]; then kill \"\$wait_pid\" 2>/dev/null || true; fi; \
        exit 130" HUP INT TERM

  echo "::group::Configuration / Setup"
  echo "Running tests in: $test_path"
  if [ -n "$pytest_options" ]; then
    echo "With pytest options: $pytest_options"
  fi
  if [ -n "$extra_pip_packages" ]; then
    echo "With extra pip packages: $extra_pip_packages"
  fi
  if [ -n "$extra_uv_packages" ]; then
    echo "With extra uv packages: $extra_uv_packages"
  fi
  if [ -n "$wheelhouse_host_dir" ]; then
    echo "With wheelhouse host directory: $wheelhouse_host_dir"
  fi
  if [ -n "$wheelhouse_packages" ]; then
    echo "With wheelhouse packages: $wheelhouse_packages"
  fi
  case "$allow_online_fallback" in
    true)
      echo "Online package-index fallback is enabled"
      ;;
    false)
      echo "Online fallback for CI wheelhouse installs is disabled"
      ;;
    *)
      echo "::error::allow-online-fallback must be 'true' or 'false', got: $allow_online_fallback"
      return 1
      ;;
  esac
  if [ -n "$filter_pattern" ]; then
    echo "With filter pattern: $filter_pattern"
  fi
  if [ -n "$exclude_pattern" ]; then
    echo "With exclude pattern: $exclude_pattern"
  fi
  if [ "$curobo_only" = "true" ]; then
    echo "cuRobo-only mode enabled: running only cuRobo and SkillGen tests"
  fi
  if [ -n "$include_files" ]; then
    echo "Include files: $include_files"
  fi
  if [ -n "$test_node_ids_file" ]; then
    echo "Test node IDs file: $test_node_ids_file"
  fi
  if [ -n "$test_node_ids_key" ]; then
    echo "Test node IDs key: $test_node_ids_key"
  fi
  if [ -n "$shard_index" ] && [ -n "$shard_count" ]; then
    echo "Shard: $shard_index of $shard_count"
  fi

  if [ -n "$test_node_ids_file" ] || [ -n "$test_node_ids_key" ]; then
    if [ -z "$test_node_ids_file" ] || [ -z "$test_node_ids_key" ]; then
      echo "Both test-node-ids-file and test-node-ids-key must be set together"
      return 1
    fi
    export TEST_NODE_IDS_FILE="$test_node_ids_file"
    export TEST_NODE_IDS_KEY="$test_node_ids_key"
  fi

  # Create reports directory
  mkdir -p "$reports_dir"

  # Clean up any existing container
  docker rm -f "$container_name" 2>/dev/null || true

  # Build Docker environment variables
  docker_env_vars="\
    -e OMNI_KIT_ACCEPT_EULA=yes \
    -e ACCEPT_EULA=Y \
    -e OMNI_KIT_DISABLE_CUP=1 \
    -e ISAAC_SIM_HEADLESS=1 \
    -e ISAAC_SIM_LOW_MEMORY=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONIOENCODING=utf-8 \
    -e GITHUB_ACTIONS=${GITHUB_ACTIONS:-} \
    -e TEST_RESULT_FILE=$result_file"

  if [ "$curobo_only" = "true" ]; then
    docker_env_vars="$docker_env_vars -e TEST_CUROBO_ONLY=true"
    echo "Setting TEST_CUROBO_ONLY=true"
  fi

  if [ "$quarantined_only" = "true" ]; then
    docker_env_vars="$docker_env_vars -e TEST_QUARANTINED_ONLY=true"
    echo "Setting TEST_QUARANTINED_ONLY=true"
  fi

  if [ -n "$include_files" ]; then
    # Strip spaces so the value is safe to embed in an unquoted docker_env_vars string.
    # conftest.py splits on commas and strips whitespace, so compact form works fine.
    include_files_compact="${include_files// /}"
    docker_env_vars="$docker_env_vars -e TEST_INCLUDE_FILES=$include_files_compact"
    echo "Setting TEST_INCLUDE_FILES=$include_files_compact"
  fi

  if [ -n "${TEST_NODE_IDS:-}" ]; then
    docker_env_vars="$docker_env_vars -e TEST_NODE_IDS"
    echo "Setting TEST_NODE_IDS"
  fi

  if [ -n "${TEST_NODE_IDS_FILE:-}" ]; then
    docker_env_vars="$docker_env_vars -e TEST_NODE_IDS_FILE -e TEST_NODE_IDS_KEY"
    echo "Setting TEST_NODE_IDS_FILE=$TEST_NODE_IDS_FILE TEST_NODE_IDS_KEY=$TEST_NODE_IDS_KEY"
  fi

  if [ -n "$shard_index" ] && [ -n "$shard_count" ]; then
    docker_env_vars="$docker_env_vars -e TEST_SHARD_INDEX=$shard_index -e TEST_SHARD_COUNT=$shard_count"
    echo "Setting TEST_SHARD_INDEX=$shard_index TEST_SHARD_COUNT=$shard_count"
  fi

  if [ -n "$filter_pattern" ]; then
    if [[ "$filter_pattern" == "not "* ]]; then
      # Handle "not <pattern>" case - note the trailing space to avoid
      # matching words that happen to start with "not".
      filter_exclude_pattern="${filter_pattern#not }"
      if [ -n "$exclude_pattern" ]; then
        exclude_pattern="${exclude_pattern},${filter_exclude_pattern}"
      else
        exclude_pattern="$filter_exclude_pattern"
      fi
    else
      # Handle positive pattern case
      docker_env_vars="$docker_env_vars -e TEST_FILTER_PATTERN=$filter_pattern"
      echo "Setting include pattern: $filter_pattern"
    fi
  else
    echo "No filter pattern provided"
  fi

  if [ -n "$exclude_pattern" ]; then
    docker_env_vars="$docker_env_vars -e TEST_EXCLUDE_PATTERN=$exclude_pattern"
    echo "Setting exclude pattern: $exclude_pattern"
  fi

  if [ -n "$extra_pip_packages" ]; then
    export TEST_EXTRA_PIP_PACKAGES="$extra_pip_packages"
    docker_env_vars="$docker_env_vars -e TEST_EXTRA_PIP_PACKAGES"
  fi
  if [ -n "$extra_uv_packages" ]; then
    export TEST_EXTRA_UV_PACKAGES="$extra_uv_packages"
    docker_env_vars="$docker_env_vars -e TEST_EXTRA_UV_PACKAGES"
  fi

  if [ -n "$test_k_expr" ]; then
    export TEST_K_EXPR="$test_k_expr"
    docker_env_vars="$docker_env_vars -e TEST_K_EXPR"
    echo "Setting per-file pytest -k expression: $test_k_expr"
  fi

  if [ -n "$ci_marker" ]; then
    docker_env_vars="$docker_env_vars -e CI_MARKER=$ci_marker"
    echo "Setting CI_MARKER=$ci_marker"
  fi

  if [ -n "$standalone_script_scope" ]; then
    docker_env_vars="$docker_env_vars \
      -e ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS=1 \
      -e ISAACLAB_STANDALONE_SCRIPT_SCOPE=$standalone_script_scope \
      -e ISAACLAB_STANDALONE_VISUALIZER=$standalone_script_visualizer"
    echo "Running standalone scripts in scope '$standalone_script_scope' with visualizer '$standalone_script_visualizer'"
    if [ -n "$standalone_script_runtime_group" ]; then
      docker_env_vars="$docker_env_vars \
        -e ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP=$standalone_script_runtime_group"
      echo "Standalone backend-runtime group: $standalone_script_runtime_group"
    fi
  fi

  export TEST_ALLOW_ONLINE_FALLBACK="$allow_online_fallback"
  docker_env_vars="$docker_env_vars -e TEST_ALLOW_ONLINE_FALLBACK"
  if [ -n "$wheelhouse_packages" ]; then
    export TEST_WHEELHOUSE_PACKAGES="$wheelhouse_packages"
    docker_env_vars="$docker_env_vars -e TEST_WHEELHOUSE_PACKAGES"
  fi

  # Volume mount for deps-cache-hit mode: bind-mount the checked-out
  # source code over /workspace/isaaclab instead of baking it into the image.
  docker_volume_args=""
  docker_user_args=""
  if [ -n "$volume_mount_source" ]; then
    host_uid="$(id -u)"
    host_gid="$(id -g)"
    host_user="$(id -un)"
    # Kit writes generated cache, config, data, and log files outside
    # the Isaac Lab source tree. Provide writable runtime storage for
    # host-uid test runs, mirroring the compose/singularity mounts.
    docker_runtime_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/isaaclab-docker-runtime.XXXXXX")"
    mkdir -p \
      "${docker_runtime_dir}/home/.cache" \
      "${docker_runtime_dir}/home/.local/share/ov/data" \
      "${docker_runtime_dir}/home/.local/share/ov/pkg" \
      "${docker_runtime_dir}/home/.nv/ComputeCache" \
      "${docker_runtime_dir}/home/.nvidia-omniverse/config" \
      "${docker_runtime_dir}/home/.nvidia-omniverse/logs" \
      "${docker_runtime_dir}/home/Documents/Kit/shared" \
      "${docker_runtime_dir}/isaac-sim/kit/cache" \
      "${docker_runtime_dir}/isaac-sim/kit/data" \
      "${docker_runtime_dir}/isaac-sim/kit/logs" \
      "${docker_runtime_dir}/isaac-sim/cache" \
      "${docker_runtime_dir}/isaac-sim/computecache" \
      "${docker_runtime_dir}/isaac-sim/config" \
      "${docker_runtime_dir}/isaac-sim/data" \
      "${docker_runtime_dir}/isaac-sim/logs" \
      "${docker_runtime_dir}/isaac-sim/pkg"
    docker_volume_args="\
      -v ${volume_mount_source}:/workspace/isaaclab:rw \
      -v ${docker_runtime_dir}/home:/tmp/isaaclab-ci-home:rw \
      -v ${docker_runtime_dir}/isaac-sim/kit/cache:/isaac-sim/kit/cache:rw \
      -v ${docker_runtime_dir}/isaac-sim/kit/data:/isaac-sim/kit/data:rw \
      -v ${docker_runtime_dir}/isaac-sim/kit/logs:/isaac-sim/kit/logs:rw \
      -v ${docker_runtime_dir}/isaac-sim/cache:/isaac-sim/.cache:rw \
      -v ${docker_runtime_dir}/isaac-sim/computecache:/isaac-sim/.nv/ComputeCache:rw \
      -v ${docker_runtime_dir}/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
      -v ${docker_runtime_dir}/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
      -v ${docker_runtime_dir}/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
      -v ${docker_runtime_dir}/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw"
    docker_user_args="--user ${host_uid}:${host_gid}"
    docker_env_vars="$docker_env_vars -e HOME=/tmp/isaaclab-ci-home -e XDG_CACHE_HOME=/tmp/isaaclab-ci-home/.cache -e XDG_DATA_HOME=/tmp/isaaclab-ci-home/.local/share -e USER=${host_user} -e LOGNAME=${host_user}"
    echo "🔵 Volume-mounting ${volume_mount_source} >> /workspace/isaaclab"
    echo "🔵 Mounting writable Docker runtime storage from ${docker_runtime_dir}"
    echo "🔵 Running volume-mounted container as host uid:gid ${host_uid}:${host_gid} (${host_user})"
  fi

  # Must follow the volume_mount_source block, which assigns rather than
  # appends to docker_volume_args. The caller creates the directory.
  if [ -n "$warp_cache_host_dir" ]; then
    docker_volume_args="$docker_volume_args \
      -v ${warp_cache_host_dir}:/tmp/isaaclab-warp-cache:rw"
    docker_env_vars="$docker_env_vars \
      -e WARP_CACHE_PATH=/tmp/isaaclab-warp-cache"
  fi

  if [ -n "$wheelhouse_host_dir" ]; then
    if [ -n "${GITHUB_ENV:-}" ]; then
      printf 'CI_WHEELHOUSE_DIR=%s\n' "${wheelhouse_host_dir}/wheelhouse" >> "$GITHUB_ENV"
      printf 'CI_WHEELHOUSE_ALLOW_ONLINE_FALLBACK=%s\n' "$allow_online_fallback" >> "$GITHUB_ENV"
    fi
    if [ ! -d "${wheelhouse_host_dir}/wheelhouse" ]; then
      wheelhouse_error="wheelhouse directory not found: ${wheelhouse_host_dir}/wheelhouse"
    elif [ ! -f "${wheelhouse_host_dir}/manifest.json" ]; then
      wheelhouse_error="wheelhouse manifest not found: ${wheelhouse_host_dir}/manifest.json"
    fi
    if [ -n "$wheelhouse_error" ]; then
      if [ "$allow_online_fallback" = "true" ]; then
        echo "::warning::${wheelhouse_error}; package installs will use configured indexes"
        wheelhouse_host_dir=""
      else
        echo "::error::${wheelhouse_error}"
        return 1
      fi
    fi
  fi

  if [ -n "$wheelhouse_host_dir" ]; then
    wheelhouse_volume_args=(-v "${wheelhouse_host_dir}:/tmp/ci-wheelhouse:ro")
    docker_env_vars="$docker_env_vars \
      -e CI_WHEELHOUSE_DIR=/tmp/ci-wheelhouse/wheelhouse \
      -e TEST_WHEELHOUSE_PATH=/tmp/ci-wheelhouse/wheelhouse \
      -e TEST_WHEELHOUSE_MANIFEST=/tmp/ci-wheelhouse/manifest.json"
    echo "Mounting CI wheelhouse read-only at /tmp/ci-wheelhouse"
  fi

  echo "Docker environment variables: '$docker_env_vars'"
  echo "::endgroup::"

  # Run tests in a detached container and follow logs.  Running detached
  # means the container lifecycle is independent of the shell - if the
  # runner kills this step on cancellation, the `if: always()` cleanup
  # step can still `docker kill` the container reliably.
  # `docker logs -f` is the foreground process and is trivially killable.
  echo "🔵 Starting Docker container for tests..."
  docker run -d --name "$container_name" \
    --init --stop-timeout 5 \
    --entrypoint bash --gpus all --network=host \
    -v "$PWD/.github/actions/_lib/with-python-package-retries.sh:/with-python-package-retries.sh:ro" \
    --security-opt=no-new-privileges:true \
    --memory="$(echo "$(free -m | awk '/^Mem:/{print $2}') * 0.9 / 1" | bc)m" \
    --cpus="$(echo "$(nproc) * 0.9" | bc)" \
    --oom-kill-disable=false \
    --ulimit nofile=65536:65536 \
    --ulimit nproc=4096:4096 \
    $docker_volume_args \
    "${wheelhouse_volume_args[@]}" \
    $docker_user_args \
    $docker_env_vars \
    "$image_tag" \
    -c "
      set -e
      cd /workspace/isaaclab
      mkdir -p tests
      rm _isaac_sim || true
      ln -s /isaac-sim _isaac_sim
      # Baseline test tooling is preinstalled in the shared CI Docker images.
      if [ -n \"\${WARP_CACHE_PATH:-}\" ]; then
        ./isaaclab.sh -p tools/verify_warp_cache.py
      fi
      if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
        if [ ! -d \"\${TEST_WHEELHOUSE_PATH}\" ]; then
          echo \"::error::CI wheelhouse path is missing: \${TEST_WHEELHOUSE_PATH}\"
          exit 1
        fi
        if [ ! -f \"\${TEST_WHEELHOUSE_MANIFEST:-}\" ]; then
          echo \"::error::CI wheelhouse manifest is missing: \${TEST_WHEELHOUSE_MANIFEST:-}\"
          exit 1
        fi
      fi
      wheelhouse_required_fallback_used=false
      if [ -n \"\${TEST_WHEELHOUSE_PACKAGES:-}\" ]; then
        read -r -a wheelhouse_packages <<< \"\${TEST_WHEELHOUSE_PACKAGES}\"
        if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
          echo \"Installing CI wheelhouse packages offline: \${TEST_WHEELHOUSE_PACKAGES}\"
          ./isaaclab.sh -p -m pip uninstall -y \"\${wheelhouse_packages[@]}\" || true
          if PIP_NO_INDEX=1 ./isaaclab.sh -p -m pip install \
            --no-index --find-links=\"\${TEST_WHEELHOUSE_PATH}\" \
            --upgrade --force-reinstall \"\${wheelhouse_packages[@]}\"; then
            wheelhouse_install_succeeded=true
          else
            wheelhouse_install_succeeded=false
          fi
        else
          wheelhouse_install_succeeded=false
        fi
        if [ \"\$wheelhouse_install_succeeded\" != \"true\" ]; then
          if [ \"\${TEST_ALLOW_ONLINE_FALLBACK:-true}\" != \"true\" ]; then
            echo \"::error::CI wheelhouse package installation failed and online fallback is disabled\"
            exit 1
          fi
          if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
            echo \"::warning::CI wheelhouse package installation failed; retrying with configured package indexes\"
          else
            echo \"::warning::CI wheelhouse is unavailable; installing requested wheelhouse packages from configured indexes\"
            ./isaaclab.sh -p -m pip uninstall -y \"\${wheelhouse_packages[@]}\" || true
          fi
          wheelhouse_required_fallback_used=true
          if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
            bash /with-python-package-retries.sh ./isaaclab.sh -p -m pip install \
              --find-links=\"\${TEST_WHEELHOUSE_PATH}\" \
              --upgrade --force-reinstall \"\${wheelhouse_packages[@]}\"
          else
            bash /with-python-package-retries.sh ./isaaclab.sh -p -m pip install \
              --upgrade --force-reinstall \"\${wheelhouse_packages[@]}\"
          fi
        fi
      fi
      if [ -n \"\${TEST_EXTRA_PIP_PACKAGES:-}\" ]; then
        read -r -a extra_pip_packages <<< \"\${TEST_EXTRA_PIP_PACKAGES}\"
        if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
          echo \"Installing extra pip packages from the CI wheelhouse: \${TEST_EXTRA_PIP_PACKAGES}\"
          if ! PIP_NO_INDEX=1 ./isaaclab.sh -p -m pip install \
            --no-index --find-links=\"\${TEST_WHEELHOUSE_PATH}\" \
            \"\${extra_pip_packages[@]}\"; then
            if [ \"\${TEST_ALLOW_ONLINE_FALLBACK:-true}\" != \"true\" ]; then
              echo \"::error::Offline extra pip package installation failed and online fallback is disabled\"
              exit 1
            fi
            echo \"::warning::Offline extra pip package installation failed; retrying with configured package indexes\"
            bash /with-python-package-retries.sh ./isaaclab.sh -p -m pip install \
              --find-links=\"\${TEST_WHEELHOUSE_PATH}\" \
              \"\${extra_pip_packages[@]}\"
          fi
        else
          echo \"Installing extra pip packages: \${TEST_EXTRA_PIP_PACKAGES}\"
          bash /with-python-package-retries.sh ./isaaclab.sh -p -m pip install \
            \"\${extra_pip_packages[@]}\"
        fi
        case \" \${TEST_EXTRA_PIP_PACKAGES} \" in
          *\" leapp\"*)
            echo \"Resolved LEAPP package:\"
            ./isaaclab.sh -p -m pip show leapp || true
            ;;
        esac
      fi
      if [ -n \"\${TEST_EXTRA_UV_PACKAGES:-}\" ]; then
        read -r -a extra_uv_packages <<< \"\${TEST_EXTRA_UV_PACKAGES}\"
        echo \"Installing extra packages with uv: \${TEST_EXTRA_UV_PACKAGES}\"
        # isaaclab.sh prints an informational line before command output, and pip
        # installs scripts into the user base when Isaac Sim site-packages is read-only.
        isaac_python=\"\$(./isaaclab.sh -p -c 'import sys; print(sys.executable)' | tail -n 1)\"
        isaac_user_site=\"\$(./isaaclab.sh -p -c 'import site; print(site.getusersitepackages())' | tail -n 1)\"
        uv_executable=\"\$(./isaaclab.sh -p -c 'import pathlib, site; print(pathlib.Path(site.getuserbase()) / \"bin\" / \"uv\")' | tail -n 1)\"
        if [ ! -x \"\${uv_executable}\" ]; then
          if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
            if ! PIP_NO_INDEX=1 ./isaaclab.sh -p -m pip install \
              --no-index --find-links=\"\${TEST_WHEELHOUSE_PATH}\" \
              --upgrade --force-reinstall uv; then
              if [ \"\${TEST_ALLOW_ONLINE_FALLBACK:-true}\" != \"true\" ]; then
                echo \"::error::Offline uv installation failed and online fallback is disabled\"
                exit 1
              fi
              echo \"::warning::Offline uv installation failed; retrying with configured package indexes\"
              bash /with-python-package-retries.sh ./isaaclab.sh -p -m pip install \
                --find-links=\"\${TEST_WHEELHOUSE_PATH}\" uv
            fi
          else
            bash /with-python-package-retries.sh ./isaaclab.sh -p -m pip install uv
          fi
        fi
        # Isaac Sim puts bundled packages ahead of the user site. Overlay only
        # the explicitly requested packages so compatible direct pins win
        # without shadowing bundled transitive dependencies such as NumPy.
        isaac_uv_overlay=\"\$(mktemp -d)\"
        install_uv_packages() {
          local use_wheelhouse=\"\$1\"
          if [ \"\$use_wheelhouse\" = \"true\" ]; then
            UV_NO_INDEX=1 UV_FIND_LINKS=\"\${TEST_WHEELHOUSE_PATH}\" \
              \"\${uv_executable}\" pip install --python \"\${isaac_python}\" \
              --target \"\${isaac_user_site}\" \"\${extra_uv_packages[@]}\" &&
            UV_NO_INDEX=1 UV_FIND_LINKS=\"\${TEST_WHEELHOUSE_PATH}\" \
              \"\${uv_executable}\" pip install --python \"\${isaac_python}\" \
              --target \"\${isaac_uv_overlay}\" --no-deps \"\${extra_uv_packages[@]}\"
          else
            if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
              UV_FIND_LINKS=\"\${TEST_WHEELHOUSE_PATH}\" \
                bash /with-python-package-retries.sh \"\${uv_executable}\" \
                pip install --python \"\${isaac_python}\" \
                --target \"\${isaac_user_site}\" \"\${extra_uv_packages[@]}\" &&
              UV_FIND_LINKS=\"\${TEST_WHEELHOUSE_PATH}\" \
                bash /with-python-package-retries.sh \"\${uv_executable}\" \
                pip install --python \"\${isaac_python}\" \
                --target \"\${isaac_uv_overlay}\" --no-deps \"\${extra_uv_packages[@]}\"
            else
              bash /with-python-package-retries.sh \"\${uv_executable}\" \
                pip install --python \"\${isaac_python}\" \
                --target \"\${isaac_user_site}\" \"\${extra_uv_packages[@]}\" &&
              bash /with-python-package-retries.sh \"\${uv_executable}\" \
                pip install --python \"\${isaac_python}\" \
                --target \"\${isaac_uv_overlay}\" --no-deps \"\${extra_uv_packages[@]}\"
            fi
          fi
        }
        if [ -n \"\${TEST_WHEELHOUSE_PATH:-}\" ]; then
          if ! install_uv_packages true; then
            if [ \"\${TEST_ALLOW_ONLINE_FALLBACK:-true}\" != \"true\" ]; then
              echo \"::error::Offline uv package installation failed and online fallback is disabled\"
              exit 1
            fi
            echo \"::warning::Offline uv package installation failed; retrying with configured package indexes\"
            rm -rf \"\${isaac_uv_overlay}\"
            isaac_uv_overlay=\"\$(mktemp -d)\"
            install_uv_packages false
          fi
        else
          install_uv_packages false
        fi
        export PYTHONPATH=\"\${isaac_uv_overlay}\${PYTHONPATH:+:\${PYTHONPATH}}\"
      fi
      if [ -n \"\${TEST_WHEELHOUSE_MANIFEST:-}\" ]; then
        TEST_WHEELHOUSE_REQUIRED_PACKAGES=\"\${TEST_WHEELHOUSE_PACKAGES:-}\" \
        TEST_WHEELHOUSE_OPTIONAL_PACKAGES=\"\${TEST_EXTRA_PIP_PACKAGES:-} \${TEST_EXTRA_UV_PACKAGES:-}\" \
        TEST_WHEELHOUSE_REQUIRED_FALLBACK_USED=\"\${wheelhouse_required_fallback_used}\" \
        ./isaaclab.sh -p tools/ci_wheelhouse/verify_installed.py
      fi
      echo 'Starting pytest with path: $test_path'
      ./isaaclab.sh -p -m pytest --ignore=tools/conftest.py $test_path $pytest_options -v --junitxml=tests/$result_file
    "

  # Stream container logs in background.
  docker logs -f "$container_name" &
  logs_pid=$!

  # Background `docker wait` + bash `wait` (interruptible by signals).
  docker wait "$container_name" > "$docker_wait_file" &
  wait_pid=$!
  wait "$wait_pid" 2>/dev/null
  local wait_status=$?
  wait_pid=""

  # If interrupted by signal, trap already handled cleanup.
  if [ "$wait_status" -gt 128 ]; then
    kill "$logs_pid" 2>/dev/null || true
    exit 130
  fi

  DOCKER_EXIT=$(cat "$docker_wait_file" 2>/dev/null) || DOCKER_EXIT=1
  rm -f "$docker_wait_file"

  # Stop following logs.
  kill "$logs_pid" 2>/dev/null || true
  wait "$logs_pid" 2>/dev/null || true
  logs_pid=""

  if [ "$DOCKER_EXIT" -eq 0 ]; then
    echo "🟢 Docker container completed successfully"
  else
    echo "🟠 Docker container failed (exit $DOCKER_EXIT), but continuing to copy results..."
  fi

  echo "::group::Copy results"
  # Copy test results with error handling.
  # When volume-mounted, test output lands on the host filesystem directly
  # (docker cp cannot see bind-mounted paths after the container stops).
  echo "🔵 Attempting to copy test results..."
  if [ -n "$volume_mount_source" ] && [ -f "${volume_mount_source}/tests/$result_file" ]; then
    cp "${volume_mount_source}/tests/$result_file" "$reports_dir/$result_file"
    echo "🟢 Test results copied from volume mount to $reports_dir/$result_file"
  elif cp_err=$(docker cp "${container_name}:/workspace/isaaclab/tests/${result_file}" "$reports_dir/$result_file" 2>&1); then
    echo "🟢 Test results copied successfully to $reports_dir/$result_file"
  else
    echo "🔴 Failed to copy specific result file: $cp_err"
    echo "🔴 Trying to copy all test results..."
    if cp_err=$(docker cp "${container_name}:/workspace/isaaclab/tests/" "$reports_dir/" 2>&1); then
      echo "🟢 All test results copied successfully to $reports_dir/"
      # Look for any XML files and use the first one found
      if [ -f "$reports_dir/full_report.xml" ]; then
        mv "$reports_dir/full_report.xml" "$reports_dir/$result_file"
        echo "🟢 Found and renamed full_report.xml to $result_file"
      elif ls "$reports_dir"/test-reports-*.xml 1>/dev/null 2>&1; then
        # Combine individual test reports if no full report exists.
        # Extract each <testsuite> block intact to produce valid JUnit XML.
        echo "🔵 Combining individual test reports..."
        echo '<?xml version="1.0" encoding="utf-8"?>' > "$reports_dir/$result_file"
        echo '<testsuites>' >> "$reports_dir/$result_file"
        for xml_file in "$reports_dir"/test-reports-*.xml; do
          if [ -f "$xml_file" ]; then
            echo "  Processing: $xml_file"
            # Copy everything between <testsuite ...> and </testsuite> (inclusive)
            sed -n '/<testsuite/,/<\/testsuite>/p' "$xml_file" >> "$reports_dir/$result_file" || echo "🟠 Warning: failed to extract testsuite from $xml_file"
          fi
        done
        echo '</testsuites>' >> "$reports_dir/$result_file"
        echo "🟢 Combined individual test reports into $result_file"
      else
        echo "🔴 No test result files found, creating fallback"
        echo "<?xml version=\"1.0\" encoding=\"utf-8\"?><testsuite name=\"$container_name\" tests=\"0\" failures=\"0\" errors=\"1\" time=\"0\"><testcase classname=\"setup\" name=\"no_results_found\"><error message=\"No test results found\">Container may have failed to generate any results</error></testcase></testsuite>" > "$reports_dir/$result_file"
      fi
    else
      echo "🔴 Failed to copy any test results, creating fallback"
      echo "<?xml version=\"1.0\" encoding=\"utf-8\"?><testsuite name=\"$container_name\" tests=\"0\" failures=\"0\" errors=\"1\" time=\"0\"><testcase classname=\"setup\" name=\"copy_failed\"><error message=\"Failed to copy test results\">Container may have failed to generate results</error></testcase></testsuite>" > "$reports_dir/$result_file"
    fi
  fi

  # Copy comparison images (saved by source/isaaclab_tasks/test/test_rendering_*.py).
  local img_dir="$reports_dir/comparison-images"
  if [ -n "$volume_mount_source" ] && [ -d "${volume_mount_source}/tests/comparison-images" ]; then
    cp -r "${volume_mount_source}/tests/comparison-images" "$img_dir"
    echo "🟢 Comparison images copied to $img_dir"
  elif docker cp "$container_name:/workspace/isaaclab/tests/comparison-images" "$img_dir" 2>/dev/null; then
    echo "🟢 Comparison images copied from container to $img_dir"
  fi
  echo "::endgroup::"

  echo "::group::Cleanup"
  # Clean up container
  echo "🔵 Cleaning up Docker container..."
  docker rm "$container_name" 2>/dev/null || echo "🟠 Container cleanup failed, but continuing..."
  if [ -n "$docker_runtime_dir" ]; then
    rm -rf "$docker_runtime_dir" || echo "🟠 Docker runtime storage cleanup failed, but continuing..."
  fi
  echo "::endgroup::"

  return "$DOCKER_EXIT"
}

# Call the function with provided parameters

run_tests "$@"
