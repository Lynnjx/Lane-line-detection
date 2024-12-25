#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32,UInt32

class ReceiveNode(Node):

    def __init__(self):
        super().__init__('receive_node')
        self.get_logger().info("开始接收" )

        self.receive_subscription = self.create_subscription(Float32,"Distance_msg",self.recv_distance_callback,7)

    def recv_distance_callback(self,distance):

        self.get_logger().info('已经收到了%.7fm距离' % distance.data)


def main(args=None):

    rclpy.init(args=args) 
    node = ReceiveNode()
    try:  
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown() 