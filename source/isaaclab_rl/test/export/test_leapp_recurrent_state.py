# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

torch = pytest.importorskip("torch")

from isaaclab_rl.leapp import is_two_tensor_lstm_state, state_dict_from_sequence, state_sequence_from_registered


def test_lstm_state_detection_requires_two_tensors():
    """Check that only two-tensor recurrent state is treated as LSTM feedback."""
    h = torch.zeros(1, 1, 4)
    c = torch.zeros(1, 1, 4)

    assert is_two_tensor_lstm_state([h, c])
    assert is_two_tensor_lstm_state((h, c))
    assert not is_two_tensor_lstm_state([h])
    assert not is_two_tensor_lstm_state([h, c, c])
    assert not is_two_tensor_lstm_state([h, object()])


def test_state_sequence_round_trip_from_dict():
    """Check named LEAPP state maps back to framework state order."""
    states = [torch.zeros(1, 1, 4), torch.ones(1, 1, 4)]
    state_dict = state_dict_from_sequence(states)

    restored = state_sequence_from_registered(state_dict, list(state_dict.keys()), states)

    assert list(state_dict.keys()) == ["actor_state_0", "actor_state_1"]
    assert restored == states
