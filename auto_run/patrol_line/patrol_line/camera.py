import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info("大家好，我是摄像头节点")
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(8)  # 打开摄像头
        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)  # 30 FPS

        if not self.cap.isOpened():
            self.get_logger().error('摄像头打开失败！')
            rclpy.shutdown()

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (176, 144))
            frame_cropped = frame_resized[44:, :, :]  # 高度从 44 到 144，只保留 100 行

            image_message = self.bridge.cv2_to_imgmsg(frame_cropped, encoding="bgr8")
            self.publisher.publish(image_message)
            self.get_logger().info('发布了一帧图像')
        else:
            self.get_logger().error('无法读取摄像头帧！')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
