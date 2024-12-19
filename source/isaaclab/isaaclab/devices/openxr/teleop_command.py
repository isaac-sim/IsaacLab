# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import carb
from omni.kit.xr.core import XRCore

TELEOP_COMMAND_EVENT_TYPE = "teleop_command"
# The event type for outgoing teleop command response text message.
TELEOP_COMMAND_RESPONSE_EVENT_TYPE = "teleop_command_response"


class Actions:
    """Class for action options."""

    START: str = "START"
    STOP: str = "STOP"
    RESET: str = "RESET"
    UNKNOWN: str = "UNKNOWN"


class TeleopCommand:
    """This class handles the message process with key word."""

    def __init__(self) -> None:
        self._message_bus = XRCore.get_singleton().get_message_bus()
        self._incoming_message_event = carb.events.type_from_string(TELEOP_COMMAND_EVENT_TYPE)
        self._outgoing_message_event = carb.events.type_from_string(TELEOP_COMMAND_RESPONSE_EVENT_TYPE)
        self._subscription = self._message_bus.create_subscription_to_pop_by_type(
            self._incoming_message_event, self._on_message
        )

    def _process_message(self, event: carb.events.IEvent):
        """Processes the received message using key word."""
        message_in = event.payload["message"]

        if "start" in message_in:
            message_out = Actions.START
        elif "stop" in message_in:
            message_out = Actions.STOP
        elif "reset" in message_in:
            message_out = Actions.RESET
        else:
            message_out = Actions.UNKNOWN

        print(f"[VC-Keyword] message_out: {message_out}")
        # Send the response back through the message bus.
        self._message_bus.push(self._outgoing_message_event, payload={"message": message_out})
        carb.log_info(f"Sent response: {message_out}")

    def _on_message(self, event: carb.events.IEvent):
        carb.log_info(f"Received message: {event.payload['message']}")
        self._process_message(event)

    def unsubscribe(self):
        self._subscription.unsubscribe()
