import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import Hobot.GPIO as GPIO
import time

class UltrasonicNode(Node):

    def __init__(self):
        super().__init__('ultrasonic_node')
        self.get_logger().info("超声波节点")
        self.distance = self.create_publisher(Float32,'Distance_msg',7)
        self.TRIG = 16
        self.ECHO = 18 
        self.distanceInit()
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.distance_detect)  

    def distanceInit(self):
        print('Distance Measurement In Progress')
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.TRIG,GPIO.OUT)
        GPIO.setup(self.ECHO,GPIO.IN)
        #GPIO.cleanup()

    def distance_detect(self):
        
        msg = Float32()
        msg.data = self.distanceStart()
        self.distance.publish(msg)  
        self.get_logger().info('探测前方距离为"%f"m' % msg.data) 
    
    def distanceStart(self):

        GPIO.output(self.TRIG,GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.TRIG,GPIO.LOW)

        while GPIO.input(self.ECHO) == 0:
            pass
        pulse_start = time.time()

        while GPIO.input(self.ECHO) == 1:
            pass
        pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 171.5
        distance = round(distance,3)
        return distance

def main(args=None):
    
    rclpy.init(args=args) 
    node = UltrasonicNode()
    try:  
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown() 
