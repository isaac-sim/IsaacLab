from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 包路径
    pkg_path = get_package_share_directory('x7_duo')
    urdf_file = os.path.join(pkg_path, 'urdf', 'x7_duo.urdf')

    # Gazebo 仿真启动（空世界）
    gazebo_pkg_path = get_package_share_directory('gazebo_ros')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg_path, 'launch', 'gazebo.launch.py')
        )
    )

    # 静态坐标变换 base_link <-> base_footprint
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_footprint_base',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint']
    )

    # 载入机器人模型到Gazebo
    spawn_model = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', urdf_file,
            '-entity', 'x7_duo'
        ],
        output='screen'
    )

    # 模拟 joint calibration (发布一个 /calibrated 话题)
    fake_joint_calibration = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/calibrated', 'std_msgs/msg/Bool', 'data: true'],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        static_tf,
        spawn_model,
        fake_joint_calibration
    ])
