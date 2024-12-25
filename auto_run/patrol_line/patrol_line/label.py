import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32  # 发布单一浮点数
from cv_bridge import CvBridge
from hobot_dnn import pyeasy_dnn as dnn
import numpy as np
import cv2

# 将 BGR 图像转换为 NV12 格式
def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))  # 转换为 NV12 格式
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

# 获取模型输入图像的高度和宽度
def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # 加载 .bin 模型
        self.models = dnn.load("/home/sunrise/Desktop/resnet18_224x224_nv12.bin")

        # 创建图像订阅器
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # 订阅话题名称
            self.image_callback,   # 图像回调函数
            10  # 队列大小
        )

        # 创建推理结果发布器
        self.result_publisher = self.create_publisher(
            Float32,  # 使用 Float32 发布单一浮点数
            '/inference/first_value',  # 发布的 ROS 话题
            10  # 队列大小
        )

        # 创建超声波距离订阅器
        self.ultrasonic_subscription = self.create_subscription(
            Float32,  # 消息类型
            'Distance_msg',  # 超声波距离话题
            self.ultrasonic_callback,  # 回调函数
            7  # 队列大小
        )
        
        # 初始化超声波数据的标志位
        self.ultrasonic_triggered = False

        # 创建 CvBridge 对象
        self.bridge = CvBridge()

        self.get_logger().info('推理节点已启动，开始订阅图像数据以及超声波数据并发布推理结果。')

    def ultrasonic_callback(self, msg):
        """处理超声波距离数据"""
        try:
            distance = msg.data
            self.get_logger().info(f'label接收到的超声波距离: {distance}')
            if distance <= 0.3:
                self.ultrasonic_triggered = True  # 设置超声波触发标志
            else:
                self.ultrasonic_triggered = False  # 距离大于0.3m，允许处理推理结果
        except Exception as e:
            self.get_logger().error(f'处理超声波数据时出错: {e}')

    def image_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 获取模型输入图像的高度和宽度
            h, w = get_hw(self.models[0].inputs[0].properties)
            des_dim = (w, h)
            resized_data = cv2.resize(frame, des_dim, interpolation=cv2.INTER_AREA)

            # 将图像转换为 NV12 格式
            nv12_data = bgr2nv12_opencv(resized_data)

            # 执行推理
            outputs = self.models[0].forward(nv12_data)
            output_array = outputs[0].buffer

            # 获取推理结果中的第一个数据
            first_value = output_array[0][0][0][0]
            
            print(f"预测坐标：{first_value}")
            
            # 确保 first_value 是浮动类型
            first_value = float(first_value)

            result_value = round(first_value,2)

            pre_x = round(result_value * 176) + 88
            cv2.line(frame, (88, 0), (88, 100), (0, 255, 0), thickness=1)
            cv2.line(frame, (pre_x, 0), (pre_x, 100), (255, 0, 0), thickness=1)
            
            img = cv2.resize(frame, (176, 100))

            if self.ultrasonic_triggered:
                status = "stop"
            else:
                status = "right" if first_value > 0.05 else "left" if first_value < -0.05 else "go straight"
                    
            text1 = f"Value: {result_value}"
            text2 = f"Status: {status}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1

            # 获取文字尺寸
            text_size1, _ = cv2.getTextSize(text1, font, font_scale, font_thickness)
            text_size2, _ = cv2.getTextSize(text2, font, font_scale, font_thickness)
            text_width = max(text_size1[0], text_size2[0])
            text_height = text_size1[1] + text_size2[1] + 10  # 两行文字的总高度加上行间距

            # 计算文字的位置，使其位于图像的中下方
            text_x = (176 - text_width) // 2
            text_y = 100 - 20  # 距离底部 20 像素

            # 绘制灰色背景矩形
            background_top_left = (text_x - 10, text_y - text_height - 10)
            background_bottom_right = (text_x + text_width + 10, text_y + 10)
            cv2.rectangle(img, background_top_left, background_bottom_right, (128, 128, 128), cv2.FILLED)

            # 在灰色背景上绘制第一行白色文字
            line1_y = text_y - text_size2[1] - 5
            cv2.putText(img, text1, (text_x, line1_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # 绘制第二行白色文字
            cv2.putText(img, text2, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # 显示推理结果
            cv2.imshow("origin", img)
            cv2.waitKey(1)  # 必须调用以刷新显示窗口

            # 发布推理结果
            self.publish_result(first_value)

        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")

    def publish_result(self, result):
        try:
            # 发布推理结果
            msg = Float32()
            msg.data = result
            self.result_publisher.publish(msg)
            self.get_logger().info(f"推理结果已发布: {result}")
        except Exception as e:
            self.get_logger().error(f"推理结果发布失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()

    try:
        # 启动节点并保持订阅
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
