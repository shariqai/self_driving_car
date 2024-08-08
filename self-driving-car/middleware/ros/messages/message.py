# ROS message definitions
import rospy
from std_msgs.msg import String

def send_message(data):
    pub = rospy.Publisher('self_driving_car_topic', String, queue_size=10)
    rospy.init_node('self_driving_car_publisher')
    pub.publish(data)
