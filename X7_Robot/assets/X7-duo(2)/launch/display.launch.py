from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 包路径
    pkg_path = get_package_share_directory('x7_duo')

    # URDF 文件路径
    urdf_file = os.path.join(pkg_path, 'urdf', 'x7_duo.urdf')

    # RViz 配置文件路径
    rviz_file = os.path.join(pkg_path, 'rviz', 'rviz_config.rviz')

    # 读取 URDF 内容作为 robot_description 参数
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # --- 各节点定义 ---
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # '-d' 指定 rviz 配置文件；移除不受支持的 '--opengl 20' 参数
        arguments=['-d', rviz_file],
        # 不再强制 LD_PRELOAD，默认环境通常更兼容。如果确实需要预加载库，请
        # 使用 env=os.environ.copy() 并合并键值后传入 Node 的 env 参数。
    )

    # --- 创建 LaunchDescription 并添加节点 ---
    ld = LaunchDescription()
    ld.add_action(joint_state_publisher_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(rviz2_node)

    return ld
