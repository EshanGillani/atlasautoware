"""
Race the AtlasAutoware stack in the AutoDRIVE Simulator (RoboRacer Sim League).

Brings up the AutoDRIVE<->AtlasAutoware bridge + the competition time-trial
controller.  Assumes the AutoDRIVE devkit bridge is already running and
connected to the simulator (separate process / container):

    # in the AutoDRIVE devkit:
    ros2 launch autodrive_roboracer bringup_headless.launch.py
    # then here:
    ros2 launch f1tenth_gym_ros autodrive_bringup_launch.py

Args:
    vehicle:=roboracer_1   AutoDRIVE namespace (legacy build: f1tenth_1)
    max_speed:=22.8        throttle=1.0 speed (m/s) — CALIBRATE to your sim
    max_steer:=0.5236      steering=1.0 angle (rad) — CALIBRATE to your sim
    steer_sign:=1.0        flip if +steer turns the wrong way (CONFIRM in sim)
    v_scale:=0.6           MPC speed cap — start low, raise per clean lap

The bridge re-publishes AutoDRIVE odom as a global pose, so raceline_mpc
tracks directly without a particle filter (sim odom is ground truth).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    for name, default in (('vehicle', 'roboracer_1'),
                          ('max_speed', '22.8'),
                          ('max_steer', '0.5236'),
                          ('steer_sign', '1.0'),
                          ('v_scale', '0.6')):
        ld.add_action(DeclareLaunchArgument(name, default_value=default))

    odom_out = '/ego_racecar/odom'

    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='autodrive_bridge',
        name='autodrive_bridge',
        output='screen',
        parameters=[{
            'vehicle': LaunchConfiguration('vehicle'),
            'max_speed': LaunchConfiguration('max_speed'),
            'max_steer': LaunchConfiguration('max_steer'),
            'steer_sign': LaunchConfiguration('steer_sign'),
            'scan_out': '/scan',
            'odom_out': odom_out,
            'imu_out': '/oakd/imu',
            'drive_in': '/drive',
        }],
    ))

    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='raceline_mpc',
        name='raceline_mpc',
        output='screen',
        parameters=[{
            'odom_topic': odom_out,
            'scan_topic': '/scan',
            'imu_topic': '/oakd/imu',
            'drive_topic': '/drive',
            'v_scale': LaunchConfiguration('v_scale'),
        }],
    ))
    return ld
