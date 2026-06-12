# Real-car bringup — sensors + actuation + the competition racing node.
#
#   rplidar_node ──► /scan ───────────┐
#   oakd_camera ──► /oakd/rgb, /oakd/imu ──► raceline_mpc ──► /drive ──► drive_node
#   (localization, e.g. particle filter, provides /pf/pose/odom separately)
#
# drive_node auto-detects its actuation path (PCA9685 over I2C, or VESC over
# UART) at startup.  Toggle individual pieces with the launch args, e.g. to
# bring up sensors only while testing on a bench:
#   ros2 launch f1tenth_gym_ros car_bringup_launch.py use_racing:=false
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'hardware.yaml',
    )
    ld = LaunchDescription()
    for arg, default in (('use_lidar', 'true'), ('use_camera', 'true'),
                         ('use_drive', 'true'), ('use_racing', 'true')):
        ld.add_action(DeclareLaunchArgument(arg, default_value=default))

    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='oakd_camera',
        name='oakd_camera',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_camera')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='drive_node',
        name='drive_node',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_drive')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='velocity_ekf',
        name='velocity_ekf',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_camera')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='particle_filter',
        name='particle_filter',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_racing')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='rc_monitor',
        name='rc_monitor',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_drive')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='supervisor',
        name='supervisor',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_drive')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='raceline_mpc',
        name='raceline_mpc',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_racing')),
    ))

    # static mounting transforms — measure on the actual car and adjust
    # args: x y z yaw pitch roll parent child
    ld.add_action(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='base_to_laser',
        arguments=['0.27', '0', '0.11', '0', '0', '0', 'base_link', 'laser'],
    ))
    ld.add_action(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='base_to_oakd',
        arguments=['0.30', '0', '0.14', '0', '0', '0', 'base_link', 'oakd_rgb'],
    ))
    return ld
