cmake_minimum_required(VERSION 3.5)
project(x7_duo)

# 使用 ament_cmake 而不是 catkin
find_package(ament_cmake REQUIRED)

# 如果你的包只包含 launch、urdf、rviz 等文件（不编译代码）
# 不需要 add_executable 或 target_link_libraries

# 安装资源文件夹
install(
  DIRECTORY launch urdf meshes config
  DESTINATION share/${PROJECT_NAME}
)

# ament 包声明
ament_package()
