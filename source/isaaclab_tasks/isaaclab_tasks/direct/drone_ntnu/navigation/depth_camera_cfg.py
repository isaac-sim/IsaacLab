
from isaaclab.utils import configclass

@configclass
class BaseDepthCameraConfig(BaseSensorConfig):
    num_sensors = 1  # number of sensors of this type

    sensor_type = "camera"  # sensor type

    # If you use more than one sensors above, there is a need to specify the sensor placement for each sensor
    # this can be added here, but the user can implement this if needed.

    # camera params VFOV is calcuated from the aspect ratio and HFOV
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)

    height = 135  # 270
    width = 240  # 480
    horizontal_fov_deg = 87.000
    max_range = 10.0
    min_range = 0.2

    # Type of camera (depth, range, pointcloud, segmentation)
    # You can combine: (depth+segmentation), (range+segmentation), (pointcloud+segmentation)
    # Other combinations are trivial and you can add support for them in the code if you want.

    calculate_depth = (
        True  # Get a depth image and not a range image. False will result in a range image
    )
    return_pointcloud = False  # Return a pointcloud instead of an image. Above depth option will be ignored if this is set to True
    pointcloud_in_world_frame = False
    segmentation_camera = True

    # transform from sensor element coordinate frame to sensor_base_link frame
    euler_frame_rot_deg = [-90.0, 0, -90.0]

    # Type of data to be returned from the sensor
    normalize_range = True  # will be set to false when pointcloud is in world frame

    # do not change this.
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # divide by max_range. Ignored when pointcloud is in world frame

    # what to do with out of range values
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0

    # randomize placement of the sensor
    randomize_placement = True
    min_translation = [0.07, -0.06, 0.01]
    max_translation = [0.12, 0.03, 0.04]
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]
    max_euler_rotation_deg = [5.0, 5.0, 5.0]

    # nominal position and orientation (only for Isaac Gym Camera Sensors)
    # If you choose to use Isaac Gym sensors, their position and orientation will NOT be randomized
    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
