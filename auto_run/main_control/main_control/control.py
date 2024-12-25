import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import serial

class ResultSubscriberNode(Node):
    def __init__(self, serial_port, baudrate):
        super().__init__('result_subscriber_node')

        # 初始化串口通信
        try:
            self.ser = serial.Serial(serial_port, baudrate=baudrate, timeout=0.5)
            self.get_logger().info(f"串口 {serial_port} 打开成功")
        except Exception as e:
            self.get_logger().error(f"串口 {serial_port} 打开失败: {e}")
            raise e

        # 创建推理结果订阅器（订阅推理结果）
        self.result_subscription = self.create_subscription(
            Float32,  # 消息类型
            '/inference/first_value',  # 推理结果话题
            self.result_callback,  # 回调函数
            10  # 队列大小
        )

        # 创建超声波距离订阅器
        self.ultrasonic_subscription = self.create_subscription(
            Float32,  # 消息类型
            'Distance_msg',  # 超声波距离话题
            self.ultrasonic_callback,  # 回调函数
            7  # 队列大小
        )

        # 初始化超声波数据的标志位以及小车速度
        self.ultrasonic_triggered = False
        self.speed = 100 

        self.get_logger().info('推理结果和超声波订阅节点已启动，正在等待数据发布。')

    def send_AP(self, angle, dis_x, dis_y):
        """通过串口发送包含角度和位置信息的数据包"""
        try:
            send_angle = int(angle + 32768)
            send_dis_x = int(dis_x + 32768)
            send_dis_y = int(dis_y + 32768)

            sendBuffer = bytearray(8)
            sendBuffer[0] = ord('#')
            sendBuffer[1] = (send_angle >> 8) & 0xff
            sendBuffer[2] = send_angle & 0xff
            sendBuffer[3] = (send_dis_x >> 8) & 0xff
            sendBuffer[4] = send_dis_x & 0xff
            sendBuffer[5] = (send_dis_y >> 8) & 0xff
            sendBuffer[6] = send_dis_y & 0xff
            sendBuffer[7] = ord('!')

            self.ser.write(sendBuffer)
            self.get_logger().info(f"发送数据: {sendBuffer.hex()}")
            self.ser.flushInput()
            self.ser.flushOutput()
        except Exception as e:
            self.get_logger().error(f"发送数据时出错: {e}")

    def result_callback(self, msg):
        """处理推理结果数据"""
        try:
            # 如果超声波检测到的距离小于等于0.3，停止处理推理结果
            if self.ultrasonic_triggered:
                return  # 跳过推理结果处理
    
            # 处理推理结果
            result_data = msg.data
            self.get_logger().info(f'接收到的原始推理结果: {result_data}')
    
            if any(abs(value) >= 0.05 for value in result_data):
                # 限制 result_data 中每个值的范围在 [-0.25, 0.25]
                result_data = [max(-0.25, min(0.25, value)) for value in result_data]

                self.speed = 100  # 转弯时速度重置为 100
                send_result = [value * 60 for value in result_data]
                self.send_AP(send_result[0], self.speed, 0)
            else:
                if self.speed < 150:
                    self.speed += 1  # 每次减速 1
                else:
                    self.speed = 150  # 最低速度限制为 150
                self.send_AP(0, self.speed, 0)

    
        except Exception as e:
            self.get_logger().error(f'处理推理结果时出错: {e}')


    def ultrasonic_callback(self, msg):
        """处理超声波距离数据"""
        try:
            distance = msg.data
            self.get_logger().info(f'接收到的超声波距离: {distance}')
            if distance <= 0.3:
                self.get_logger().info('超声波检测到距离为0.3m，停止发送指令')
                self.send_AP(0, 0, 0)
                self.ultrasonic_triggered = True  # 设置超声波触发标志
            else:
                self.ultrasonic_triggered = False  # 距离大于0.3m，允许处理推理结果
                self.get_logger().info('继续发送指令')
        except Exception as e:
            self.get_logger().error(f'处理超声波数据时出错: {e}')

    def destroy_node(self):
        # 关闭串口
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("串口已关闭")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # 设置串口参数
    serial_port = '/dev/ttyUSB0'  # 修改为实际使用的串口
    baudrate = 115200

    try:
        node = ResultSubscriberNode(serial_port, baudrate)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
