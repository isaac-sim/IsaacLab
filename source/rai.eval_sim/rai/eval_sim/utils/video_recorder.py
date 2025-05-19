# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os

import cv2
from isaaclab.utils import configclass
from isaacsim.core.api.simulation_context import SimulationContext
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.widget.viewport.capture import FileCapture
from rai.eval_sim.utils import log_info, log_warn

FILEPATH_TIME_FORMATTING = "%Y-%m-%d_%H-%M-%S%f"


@configclass
class VideoRecorderCfg:
    """Configuration parameters for a video recorder."""

    target_framerate: int = 20
    """Target recording framerate in frames per second.

    The reason this is a target is because the actual framerate may be lower if the simulation is too slow.
    """
    video_frames_path: str = os.path.join(os.getenv("ISAACLAB_PATH"), "videos/tempframes/")
    """Directory to save frame images that will be used to create the video."""
    video_path: str = os.path.join(os.getenv("ISAACLAB_PATH"), "videos/")
    """Directory to save video to.

    NOTE: The video's file name will be the time at which the recording was stopped.
    """
    add_timing_overlay: bool = True
    """Whether to add a timing overlay to the video showing timestamp, fps, and real-time %."""


class VideoRecorder:
    def __init__(self, sim: SimulationContext, cfg: VideoRecorderCfg):
        self._sim = sim
        self.cfg = cfg
        self._viewport_api = None
        self._recording: bool = False
        self._frame_times: list = []

    @property
    def recording(self):
        return self._recording

    def start_recording(self):
        log_info("Starting video recording.")
        self.reset()
        self._recording = True
        self._frame_times = [datetime.datetime.now()]

    def stop_recording(self):
        log_info("Stopping video recording.")
        self._recording = False
        self._save_video()

    def _save_video(self):
        if len(self._frame_times) > 1:
            log_info(f"Saving video to {self.cfg.video_path}.")
            # Define the video codec and frame rate
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self._calculate_framerate()

            # Get all image paths in the directory and sort them to get them in order of capture
            image_paths = [img for img in os.listdir(self.cfg.video_frames_path) if img.endswith(".png")]
            image_paths.sort()

            # Define the frame size based on the first image
            first_frame = cv2.imread(os.path.join(self.cfg.video_frames_path, image_paths[0]))
            height, width, _ = first_frame.shape

            # Make filename based on the last frame's timestamp
            video_filename = os.path.join(
                self.cfg.video_path, f"{self._frame_times[-1].strftime(FILEPATH_TIME_FORMATTING)}.mp4"
            )

            video = cv2.VideoWriter(filename=video_filename, fourcc=fourcc, fps=fps, frameSize=(width, height))

            # Loop through each image and write it to the video
            for i, image_path in enumerate(image_paths):
                curr_image = cv2.imread(os.path.join(self.cfg.video_frames_path, image_path))

                if self.cfg.add_timing_overlay:
                    # Convert timestamp to datetime object
                    datetime_obj = datetime.datetime.strptime(image_path[:-4], FILEPATH_TIME_FORMATTING)

                    # Convert datetime object to human-readable format
                    image_text = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")

                    # Text is added to the image to display the time at which the image was captured time at which it was taken
                    curr_image = cv2.putText(
                        curr_image, image_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )

                    expected_time_diff = 1.0 / self.cfg.target_framerate
                    if i == 0:
                        prev_datetime_obj = datetime_obj
                    else:
                        # We can only add the FPS and perecentage FPS if we have at least 2 frames
                        time_diff = datetime_obj - prev_datetime_obj
                        time_diff_seconds = time_diff.total_seconds()

                        percentage_fps = (1.0 / (time_diff_seconds / expected_time_diff)) * 100.0

                        image_text += f" | FPS: {(1.0 / time_diff_seconds):.1f}".rjust(7)
                        image_text += f" | % Target FPS: {percentage_fps:.1f} %".rjust(7)

                    curr_image = cv2.putText(
                        curr_image, image_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )

                video.write(curr_image)

            # Release the VideoWriter object
            video.release()
            log_info(f"Video saved successfully to: {video_filename}")
        else:
            log_warn("No frames to save to video. Make sure simulation is playing before stopping video recording.")

    def reset(self):
        if os.path.exists(self.cfg.video_frames_path):
            log_warn(f"Deleting all files in {self.cfg.video_frames_path} from previous video recording.")
            for f in os.listdir(self.cfg.video_frames_path):
                os.remove(os.path.join(self.cfg.video_frames_path, f))
        else:
            log_warn(f"Creating directory {self.cfg.video_frames_path} for video frames.")
            os.makedirs(self.cfg.video_frames_path)

    def step(self):
        """Record a frame from the viewport if it is time to do so."""
        if self._do_record():
            self._record_frame()

    def _record_frame(self):
        """Record a frame from the viewport.

        The frame will be saved to disk with the current timestamp as the filename.
        """
        if self._viewport_api is None:
            self._viewport_api = get_active_viewport()

        current_time = datetime.datetime.now()

        self._viewport_api.schedule_capture(
            FileCapture(
                os.path.join(self.cfg.video_frames_path, f"{current_time.strftime(FILEPATH_TIME_FORMATTING)}.png"),
            )
        )
        self._frame_times.append(current_time)

    def _do_record(self) -> bool:
        """Check if it is time to record a frame.

        This is based upon whether the time since the last frame is greater than 1 / target framerate.

        Returns:
            bool: Whether it is time to record a frame based upon the framerate.
        """
        if self.recording:
            time_since_last_frame = datetime.datetime.now() - self._frame_times[-1]

            # Warning if time between frames is greater than 1.5x the targeted time
            if time_since_last_frame > datetime.timedelta(seconds=1.5 / self.cfg.target_framerate):
                log_warn(
                    f"Target framerate of {self.cfg.target_framerate} fps not met.\nMeasured framerate:"
                    f" {1.0 / time_since_last_frame.total_seconds()} fps."
                )

            return time_since_last_frame.total_seconds() >= (1.0 / self.cfg.target_framerate)
        else:
            return False

    def _calculate_framerate(self) -> float:
        """Calculate the framerate (in fps) of the video.

        Averages the time between frames.

        Returns:
            float: The average framerate of the video (in frames per second).
        """
        if len(self._frame_times) < 2:
            return self.cfg.target_framerate

        time_diffs = [
            (self._frame_times[i] - self._frame_times[i - 1]).total_seconds() for i in range(1, len(self._frame_times))
        ]

        # Calculate the average time between frames
        avg = sum(time_diffs) / len(time_diffs)

        # Calculate the framerate
        framerate = 1 / avg

        return framerate
