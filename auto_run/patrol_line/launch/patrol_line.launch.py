from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_node = Node(
        package="patrol_line",
        executable="camera_node"
        )

    label_node = Node(
        package="patrol_line",
        executable="label_node"
        )
        
    control_node = Node(
        package="main_control",
        executable="control_node"
        )
    
    ultrasonic_node = Node(
        package="avoid_obstacle",
        executable="ultrasonic_node"
        )
    launch_description = LaunchDescription([camera_node,label_node,control_node,ultrasonic_node])
    return launch_description
