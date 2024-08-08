# ROS service code
import rospy
from std_srvs.srv import Empty, EmptyResponse

def handle_service(req):
    return EmptyResponse()

def start_service():
    rospy.init_node('self_driving_car_service')
    s = rospy.Service('self_driving_car_service', Empty, handle_service)
    rospy.spin()
