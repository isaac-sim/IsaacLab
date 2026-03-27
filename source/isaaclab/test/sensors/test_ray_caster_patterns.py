# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import pytest
import torch

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=False).app

# Import after app launch
from isaaclab.sensors.ray_caster.patterns import patterns, patterns_cfg


@pytest.fixture(scope="module", params=["cuda", "cpu"])
def device(request):
    """Fixture to parameterize tests over both CUDA and CPU devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


class TestGridPattern:
    """Test cases for grid_pattern function."""

    @pytest.mark.parametrize(
        "size,resolution,ordering,expected_num_rays",
        [
            ((2.0, 2.0), 1.0, "xy", 9),  # 3x3 grid
            ((2.0, 2.0), 0.5, "xy", 25),  # 5x5 grid
            ((4.0, 2.0), 1.0, "xy", 15),  # 5x3 grid
            ((2.0, 4.0), 1.0, "yx", 15),  # 3x5 grid
            ((1.0, 1.0), 0.25, "xy", 25),  # 5x5 grid with smaller size
        ],
    )
    def test_grid_pattern_num_rays(self, device, size, resolution, ordering, expected_num_rays):
        """Test that grid pattern generates the correct number of rays."""
        cfg = patterns_cfg.GridPatternCfg(size=size, resolution=resolution, ordering=ordering)
        ray_starts, ray_directions = patterns.grid_pattern(cfg, device)

        assert ray_starts.shape[0] == expected_num_rays
        assert ray_directions.shape[0] == expected_num_rays
        assert ray_starts.shape[1] == 3
        assert ray_directions.shape[1] == 3

    @pytest.mark.parametrize("ordering", ["xy", "yx"])
    def test_grid_pattern_ordering(self, device, ordering):
        """Test that grid pattern respects the ordering parameter."""
        cfg = patterns_cfg.GridPatternCfg(size=(2.0, 2.0), resolution=1.0, ordering=ordering)
        ray_starts, ray_directions = patterns.grid_pattern(cfg, device)

        # Check that the rays are ordered correctly
        if ordering == "xy":
            # For "xy" ordering, x should change faster than y
            # First few rays should have same y, different x
            assert ray_starts[0, 1] == ray_starts[1, 1]  # Same y
            assert ray_starts[0, 0] != ray_starts[1, 0]  # Different x
        else:  # "yx"
            # For "yx" ordering, y should change faster than x
            # First few rays should have same x, different y
            assert ray_starts[0, 0] == ray_starts[1, 0]  # Same x
            assert ray_starts[0, 1] != ray_starts[1, 1]  # Different y

    @pytest.mark.parametrize("direction", [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
    def test_grid_pattern_direction(self, device, direction):
        """Test that grid pattern uses the specified direction."""
        cfg = patterns_cfg.GridPatternCfg(size=(2.0, 2.0), resolution=1.0, direction=direction)
        ray_starts, ray_directions = patterns.grid_pattern(cfg, device)

        expected_direction = torch.tensor(direction, device=device)
        # All rays should have the same direction - check in batch
        expected_directions = expected_direction.unsqueeze(0).expand_as(ray_directions)
        torch.testing.assert_close(ray_directions, expected_directions)

    def test_grid_pattern_bounds(self, device):
        """Test that grid pattern respects the size bounds."""
        size = (2.0, 4.0)
        cfg = patterns_cfg.GridPatternCfg(size=size, resolution=1.0)
        ray_starts, ray_directions = patterns.grid_pattern(cfg, device)

        # Check that all rays are within bounds
        assert ray_starts[:, 0].min() >= -size[0] / 2
        assert ray_starts[:, 0].max() <= size[0] / 2
        assert ray_starts[:, 1].min() >= -size[1] / 2
        assert ray_starts[:, 1].max() <= size[1] / 2
        # Z should be 0 for grid pattern
        torch.testing.assert_close(ray_starts[:, 2], torch.zeros_like(ray_starts[:, 2]))

    def test_grid_pattern_invalid_ordering(self, device):
        """Test that invalid ordering raises ValueError."""
        cfg = patterns_cfg.GridPatternCfg(size=(2.0, 2.0), resolution=1.0, ordering="invalid")
        with pytest.raises(ValueError, match="Ordering must be 'xy' or 'yx'"):
            patterns.grid_pattern(cfg, device)

    def test_grid_pattern_invalid_resolution(self, device):
        """Test that invalid resolution raises ValueError."""
        cfg = patterns_cfg.GridPatternCfg(size=(2.0, 2.0), resolution=-1.0)
        with pytest.raises(ValueError, match="Resolution must be greater than 0"):
            patterns.grid_pattern(cfg, device)


class TestLidarPattern:
    """Test cases for lidar_pattern function."""

    @pytest.mark.parametrize(
        "horizontal_fov_range,horizontal_res,channels,vertical_fov_range",
        [
            # Test 360 degree horizontal FOV
            ((-180.0, 180.0), 90.0, 1, (-10.0, -10.0)),
            ((-180.0, 180.0), 45.0, 1, (-10.0, -10.0)),
            ((-180.0, 180.0), 1.0, 1, (-10.0, -10.0)),
            # Test partial horizontal FOV
            ((-90.0, 90.0), 30.0, 1, (-10.0, -10.0)),
            ((0.0, 180.0), 45.0, 1, (-10.0, -10.0)),
            # Test 360 no overlap case
            ((-180.0, 180.0), 90.0, 1, (0.0, 0.0)),
            # Test partial FOV case
            ((-90.0, 90.0), 90.0, 1, (0.0, 0.0)),
            # Test multiple channels
            ((-180.0, 180.0), 90.0, 16, (-15.0, 15.0)),
            ((-180.0, 180.0), 45.0, 32, (-30.0, 10.0)),
            # Test single channel, different vertical angles
            ((-180.0, 180.0), 90.0, 1, (45.0, 45.0)),
        ],
    )
    def test_lidar_pattern_num_rays(self, device, horizontal_fov_range, horizontal_res, channels, vertical_fov_range):
        """Test that lidar pattern generates the correct number of rays."""
        cfg = patterns_cfg.LidarPatternCfg(
            horizontal_fov_range=horizontal_fov_range,
            horizontal_res=horizontal_res,
            channels=channels,
            vertical_fov_range=vertical_fov_range,
        )
        ray_starts, ray_directions = patterns.lidar_pattern(cfg, device)

        # Calculate expected number of horizontal angles
        if abs(abs(horizontal_fov_range[0] - horizontal_fov_range[1]) - 360.0) < 1e-6:
            # 360 degree FOV - exclude last point to avoid overlap
            expected_num_horizontal = (
                math.ceil((horizontal_fov_range[1] - horizontal_fov_range[0]) / horizontal_res) + 1
            ) - 1
        else:
            expected_num_horizontal = (
                math.ceil((horizontal_fov_range[1] - horizontal_fov_range[0]) / horizontal_res) + 1
            )

        expected_num_rays = channels * expected_num_horizontal

        assert ray_starts.shape[0] == expected_num_rays, (
            f"Expected {expected_num_rays} rays, got {ray_starts.shape[0]} rays. "
            f"Horizontal angles: {expected_num_horizontal}, channels: {channels}"
        )
        assert ray_directions.shape[0] == expected_num_rays
        assert ray_starts.shape[1] == 3
        assert ray_directions.shape[1] == 3

    def test_lidar_pattern_basic_properties(self, device):
        """Test that ray directions are normalized and rays start from origin."""
        cfg = patterns_cfg.LidarPatternCfg(
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=45.0,
            channels=8,
            vertical_fov_range=(-15.0, 15.0),
        )
        ray_starts, ray_directions = patterns.lidar_pattern(cfg, device)

        # Check that all directions are unit vectors
        norms = torch.norm(ray_directions, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

        # All rays should start from origin
        torch.testing.assert_close(ray_starts, torch.zeros_like(ray_starts))

    @pytest.mark.parametrize(
        "vertical_fov_range,channels",
        [
            ((-15.0, 15.0), 4),
            ((-30.0, 10.0), 5),
            ((0.0, 0.0), 1),
        ],
    )
    def test_lidar_pattern_vertical_channels(self, device, vertical_fov_range, channels):
        """Test that vertical channels are distributed correctly."""
        cfg = patterns_cfg.LidarPatternCfg(
            horizontal_fov_range=(0.0, 0.0),  # Single horizontal direction
            horizontal_res=1.0,
            channels=channels,
            vertical_fov_range=vertical_fov_range,
        )
        ray_starts, ray_directions = patterns.lidar_pattern(cfg, device)

        assert ray_starts.shape[0] == channels

        # Check that z-components span the vertical range
        # For single horizontal direction (0,0), the z component is sin(vertical_angle)
        z_components = ray_directions[:, 2]
        expected_min_z = math.sin(math.radians(vertical_fov_range[0]))
        expected_max_z = math.sin(math.radians(vertical_fov_range[1]))

        assert torch.isclose(z_components.min(), torch.tensor(expected_min_z, device=device), atol=1e-5)
        assert torch.isclose(z_components.max(), torch.tensor(expected_max_z, device=device), atol=1e-5)

    @pytest.mark.parametrize(
        "horizontal_fov_range,horizontal_res,expected_num_rays,expected_angular_spacing",
        [
            # Test case from the bug fix: 360 deg FOV with 90 deg resolution
            ((-180.0, 180.0), 90.0, 4, 90.0),
            # Test case: 360 deg FOV with 45 deg resolution
            ((-180.0, 180.0), 45.0, 8, 45.0),
            # Test case: 180 deg FOV with 90 deg resolution
            ((-90.0, 90.0), 90.0, 3, 90.0),
            # Test case: 180 deg FOV with 60 deg resolution (avoids atan2 discontinuity at ±180°)
            ((-90.0, 90.0), 60.0, 4, 60.0),
            # Test case: 360 deg FOV with 120 deg resolution
            ((-180.0, 180.0), 120.0, 3, 120.0),
        ],
    )
    def test_lidar_pattern_exact_angles(
        self, device, horizontal_fov_range, horizontal_res, expected_num_rays, expected_angular_spacing
    ):
        """Test that lidar pattern generates rays with correct count and angular spacing.

        This test verifies the fix for the horizontal angle calculation to ensure
        the actual resolution matches the requested resolution.
        """
        cfg = patterns_cfg.LidarPatternCfg(
            horizontal_fov_range=horizontal_fov_range,
            horizontal_res=horizontal_res,
            channels=1,
            vertical_fov_range=(0.0, 0.0),
        )
        ray_starts, ray_directions = patterns.lidar_pattern(cfg, device)

        # Check that we have the right number of rays
        assert ray_starts.shape[0] == expected_num_rays, (
            f"Expected {expected_num_rays} rays, got {ray_starts.shape[0]} rays"
        )

        # Calculate angles from directions
        angles = torch.atan2(ray_directions[:, 1], ray_directions[:, 0])
        angles_deg = torch.rad2deg(angles)

        # Sort angles for easier checking
        angles_deg_sorted = torch.sort(angles_deg)[0]

        # Check angular spacing between consecutive rays
        for i in range(len(angles_deg_sorted) - 1):
            angular_diff = abs(angles_deg_sorted[i + 1].item() - angles_deg_sorted[i].item())
            # Allow small tolerance for floating point errors
            assert abs(angular_diff - expected_angular_spacing) < 1.0, (
                f"Angular spacing {angular_diff:.2f}° does not match expected {expected_angular_spacing}°"
            )

        # For 360 degree FOV, also check that first and last angles wrap correctly
        is_360 = abs(abs(horizontal_fov_range[0] - horizontal_fov_range[1]) - 360.0) < 1e-6
        if is_360:
            # The gap from last angle back to first angle (wrapping around) should also match spacing
            first_angle = angles_deg_sorted[0].item()
            last_angle = angles_deg_sorted[-1].item()
            wraparound_diff = (first_angle + 360) - last_angle
            assert abs(wraparound_diff - expected_angular_spacing) < 1.0, (
                f"Wraparound spacing {wraparound_diff:.2f}° does not match expected {expected_angular_spacing}°"
            )


class TestBpearlPattern:
    """Test cases for bpearl_pattern function."""

    @pytest.mark.parametrize(
        "horizontal_fov,horizontal_res",
        [
            (360.0, 10.0),  # Default config
            (360.0, 5.0),
            (180.0, 10.0),
            (90.0, 5.0),
        ],
    )
    def test_bpearl_pattern_horizontal_params(self, device, horizontal_fov, horizontal_res):
        """Test bpearl pattern with different horizontal parameters."""
        cfg = patterns_cfg.BpearlPatternCfg(
            horizontal_fov=horizontal_fov,
            horizontal_res=horizontal_res,
        )
        ray_starts, ray_directions = patterns.bpearl_pattern(cfg, device)

        # Calculate expected number of horizontal angles
        expected_num_horizontal = int(horizontal_fov / horizontal_res)
        expected_num_rays = len(cfg.vertical_ray_angles) * expected_num_horizontal

        assert ray_starts.shape[0] == expected_num_rays

    def test_bpearl_pattern_basic_properties(self, device):
        """Test that ray directions are normalized and rays start from origin."""
        cfg = patterns_cfg.BpearlPatternCfg()
        ray_starts, ray_directions = patterns.bpearl_pattern(cfg, device)

        # Check that all directions are unit vectors
        norms = torch.norm(ray_directions, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

        # All rays should start from origin
        torch.testing.assert_close(ray_starts, torch.zeros_like(ray_starts))

    def test_bpearl_pattern_custom_vertical_angles(self, device):
        """Test bpearl pattern with custom vertical angles."""
        custom_angles = [10.0, 20.0, 30.0, 40.0, 50.0]
        cfg = patterns_cfg.BpearlPatternCfg(
            horizontal_fov=360.0,
            horizontal_res=90.0,
            vertical_ray_angles=custom_angles,
        )
        ray_starts, ray_directions = patterns.bpearl_pattern(cfg, device)

        # 360/90 = 4 horizontal angles, 5 custom vertical angles
        expected_num_rays = 4 * 5
        assert ray_starts.shape[0] == expected_num_rays


class TestPinholeCameraPattern:
    """Test cases for pinhole_camera_pattern function."""

    @pytest.mark.parametrize(
        "width,height",
        [
            (640, 480),
            (1920, 1080),
            (320, 240),
            (100, 100),
        ],
    )
    def test_pinhole_camera_pattern_num_rays(self, device, width, height):
        """Test that pinhole camera pattern generates the correct number of rays."""
        cfg = patterns_cfg.PinholeCameraPatternCfg(
            width=width,
            height=height,
        )

        # Create a simple intrinsic matrix for testing
        # Using identity-like matrix with focal lengths and principal point at center
        fx = fy = 500.0
        cx = width / 2
        cy = height / 2
        intrinsic_matrix = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=device,
        )

        # Pattern expects batch of intrinsic matrices
        intrinsic_matrices = intrinsic_matrix.unsqueeze(0)

        ray_starts, ray_directions = patterns.pinhole_camera_pattern(cfg, intrinsic_matrices, device)

        expected_num_rays = width * height
        assert ray_starts.shape == (1, expected_num_rays, 3)
        assert ray_directions.shape == (1, expected_num_rays, 3)

    def test_pinhole_camera_pattern_basic_properties(self, device):
        """Test that ray directions are normalized and rays start from origin."""
        cfg = patterns_cfg.PinholeCameraPatternCfg(width=100, height=100)

        fx = fy = 500.0
        cx = cy = 50.0
        intrinsic_matrix = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=device,
        ).unsqueeze(0)

        ray_starts, ray_directions = patterns.pinhole_camera_pattern(cfg, intrinsic_matrix, device)

        # Check that all directions are unit vectors
        norms = torch.norm(ray_directions, dim=2)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

        # All rays should start from origin
        torch.testing.assert_close(ray_starts, torch.zeros_like(ray_starts))

    def test_pinhole_camera_pattern_batch(self, device):
        """Test that pinhole camera pattern works with batched intrinsic matrices."""
        cfg = patterns_cfg.PinholeCameraPatternCfg(width=50, height=50)

        # Create batch of 3 different intrinsic matrices
        batch_size = 3
        intrinsic_matrices = []
        for i in range(batch_size):
            fx = fy = 500.0 + i * 100
            cx = cy = 25.0
            intrinsic_matrices.append(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device))
        intrinsic_matrices = torch.stack(intrinsic_matrices)

        ray_starts, ray_directions = patterns.pinhole_camera_pattern(cfg, intrinsic_matrices, device)

        expected_num_rays = 50 * 50
        assert ray_starts.shape == (batch_size, expected_num_rays, 3)
        assert ray_directions.shape == (batch_size, expected_num_rays, 3)

        # Check that different batches have different ray directions (due to different intrinsics)
        assert not torch.allclose(ray_directions[0], ray_directions[1])

    def test_pinhole_camera_from_intrinsic_matrix(self, device):
        """Test creating PinholeCameraPatternCfg from intrinsic matrix."""
        width, height = 640, 480
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0

        intrinsic_list = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

        cfg = patterns_cfg.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_list,
            width=width,
            height=height,
        )

        assert cfg.width == width
        assert cfg.height == height
        assert cfg.focal_length == 24.0  # default

        # The apertures should be calculated based on the intrinsic matrix
        assert cfg.horizontal_aperture > 0
        assert cfg.vertical_aperture > 0
