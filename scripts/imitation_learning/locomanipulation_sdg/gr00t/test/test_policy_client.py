# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the client against N1.5's actual server in a separate interpreter.

Set GR00T_PYTHON to the isolated N1.5 interpreter. Run pytest in a client
environment containing NumPy 2, msgpack and pyzmq, without GR00T installed.
The server handler validates a small observation and returns known actions;
no model download or GPU is needed to test the upstream wire contract.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_client import PolicyClient


@pytest.fixture
def upstream_server(tmp_path):
    python = os.environ.get("GR00T_PYTHON")
    if not python:
        pytest.skip("Set GR00T_PYTHON to the isolated N1.5 interpreter")
    port_file = tmp_path / "port"
    # Bind port 0 and discover the assigned port in the child to avoid a port-selection race.
    script = """
import sys
import time
from pathlib import Path
import numpy as np
import zmq
from gr00t.eval.service import BaseInferenceServer

def get_action(observation):
    if observation.get('fail'):
        raise ValueError('invalid observation')
    if observation.get('slow'):
        time.sleep(0.2)
    image = observation['video.ego_view']
    state = observation['state.base_height']
    assert image.dtype == np.uint8 and image.shape == (1, 2, 3, 3)
    assert image[0, 1, 2].tolist() == [15, 16, 17]
    assert state.dtype == np.float32 and state.tolist() == [[0.75]]
    return {'action.base_height': np.full((16, 1), 0.8, dtype=np.float32)}

server = BaseInferenceServer(host='127.0.0.1', port=0)
server.register_endpoint('get_action', get_action)
Path(sys.argv[1]).write_text(server.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(':', 1)[1])
server.run()
"""
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([python, "-u", "-c", script, str(port_file)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while not port_file.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"N1.5 server did not start: {log_path.read_text()}")
                time.sleep(0.05)
            yield int(port_file.read_text())
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def test_upstream_arrays_errors_and_timeout_recovery(upstream_server):
    """NumPy 2 observations work on N1.5; failures do not poison subsequent requests."""
    observations = {
        "video.ego_view": np.arange(18, dtype=np.uint8).reshape(1, 2, 3, 3),
        "state.base_height": np.array([[0.75]], dtype=np.float32),
    }
    with PolicyClient(port=upstream_server, timeout_ms=1000) as client:
        client.ping()
        actions = client.get_action(observations)
        assert set(actions) == {"action.base_height"}
        assert actions["action.base_height"].shape == (16, 1)
        assert actions["action.base_height"].dtype == np.float32
        np.testing.assert_allclose(actions["action.base_height"], 0.8)
        with pytest.raises(RuntimeError, match="invalid observation"):
            client.get_action({"fail": True})
        client.ping()
        client.timeout_ms = 50
        with pytest.raises(TimeoutError, match="did not respond"):
            client.get_action({**observations, "slow": True})
        client.timeout_ms = 1000
        client.ping()


def test_unavailable_server_has_bounded_timeout():
    """A socket that accepts no requests cannot hang rollout or client shutdown."""
    with zmq.Context() as context, context.socket(zmq.REP) as silent_server:
        port = silent_server.bind_to_random_port("tcp://127.0.0.1")
        started = time.monotonic()
        with PolicyClient(port=port, timeout_ms=50) as client:
            with pytest.raises(TimeoutError, match="serve_policy.py"):
                client.ping()
        assert time.monotonic() - started < 2
