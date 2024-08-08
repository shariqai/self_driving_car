# Camera driver code
import cv2

def start_camera():
    # Implement starting of camera driver
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not start camera")
    return cap
