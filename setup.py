from setuptools import setup
import os
from glob import glob
package_name = 'f1tenth_gym_ros'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.rl'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xacro')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Billy Zheng',
    maintainer_email='billyzheng.bz@gmail.com',
    description='Bridge for using f1tenth_gym in ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gym_bridge = f1tenth_gym_ros.gym_bridge:main',
            'racing_agent = f1tenth_gym_ros.pursuit_agent:main',
            'race_agent = f1tenth_gym_ros.race_agent:main',
            'opponent_driver = f1tenth_gym_ros.opponent_driver:main',
            'mapping_driver = f1tenth_gym_ros.mapping_driver:main',
            'track_learner = f1tenth_gym_ros.track_learner:main',
            'raceline_mpc = f1tenth_gym_ros.raceline_mpc:main',
            'rl_agent = f1tenth_gym_ros.rl_agent:main',
            'camera_perception = f1tenth_gym_ros.camera_perception:main',
            'drive_node = f1tenth_gym_ros.drive_node:main',
            'rplidar_node = f1tenth_gym_ros.rplidar_node:main',
            'oakd_camera = f1tenth_gym_ros.oakd_camera:main',
            'velocity_ekf = f1tenth_gym_ros.velocity_ekf:main',
        ],
    },
)
 