import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='port_name',
                                             default_value='ttyTHS1'),
        launch.actions.DeclareLaunchArgument(name='odom_topic_name',
                                             default_value='odom'),
        launch.actions.DeclareLaunchArgument(name='open_rviz',
                                             default_value='false'),
        launch.actions.DeclareLaunchArgument(name='camera_device',
                                             default_value='/dev/video0',
                                             description='Camera device path'),
        launch.actions.DeclareLaunchArgument(name='image_topic',
                                             default_value='image',
                                             description='Image topic name'),
        launch.actions.DeclareLaunchArgument(name='camera_info_topic',
                                             default_value='camera_info',
                                             description='Camera info topic name'),
        launch.actions.DeclareLaunchArgument(name='frame_id',
                                             default_value='camera_frame',
                                             description='Frame ID for camera'),
        launch.actions.DeclareLaunchArgument(name='width',
                                             default_value='640',
                                             description='Image width'),
        launch.actions.DeclareLaunchArgument(name='height',
                                             default_value='480',
                                             description='Image height'),
        launch.actions.DeclareLaunchArgument(name='frequency',
                                             default_value='30.0',
                                             description='Camera frequency'),
        
        launch_ros.actions.Node(
            package='rviz2',
            name='rviz2',
            executable='rviz2',
            on_exit=launch.actions.Shutdown(),
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('open_rviz'))),
        
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_imu',
            arguments="0.0 0.0 0.0 0.0 0.0 0.0 /base_link /imu_link".split(' ')),
        
        # Static transform from base_link to camera_frame
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_camera',
            arguments="0.1 0.0 0.1 0.0 0.0 0.0 /base_link /camera_frame".split(' ')),
        
        # Camera node using image_tools cam2image
        launch_ros.actions.Node(
            package='image_tools',
            executable='cam2image',
            name='camera_node',
            parameters=[{
                'device': launch.substitutions.LaunchConfiguration('camera_device'),
                'width': launch.substitutions.LaunchConfiguration('width'),
                'height': launch.substitutions.LaunchConfiguration('height'),
                'frequency': launch.substitutions.LaunchConfiguration('frequency'),
                'frame_id': launch.substitutions.LaunchConfiguration('frame_id'),
            }],
            remappings=[
                ('image', launch.substitutions.LaunchConfiguration('image_topic')),
                ('camera_info', launch.substitutions.LaunchConfiguration('camera_info_topic')),
            ],
            arguments=['--ros-args', '--log-level', 'WARN']),
        
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('limo_base'),
                             'launch/limo_base.launch.py')),
            launch_arguments={
                'port_name':
                launch.substitutions.LaunchConfiguration('port_name'),
                'odom_topic_name':
                launch.substitutions.LaunchConfiguration('odom_topic_name')
            }.items()),
        
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('limo_base'),
                             'launch','open_ydlidar_launch.py')))
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
