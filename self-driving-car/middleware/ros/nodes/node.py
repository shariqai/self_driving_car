# ROS node code
import rospy

def start_node():
    rospy.init_node('self_driving_car_node')
    rospy.spin()
