#!/usr/bin/env python

from typing import Dict
import roslib
import sys
import rospy
import cv2
import numpy as np
import Process
import json
import BenchOS.GPSVision.image_manipulation
from std_msgs.msg import String

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
      
    self.robotPosition = None
    # initialize the node named image_processing
    rospy.init_node('GPSVideo_Processor', anonymous=True)
    # initialize a publisher to send xz coordinates
    self.pos_pub = rospy.Publisher("robot_position", String ,queue_size = 1)
    
    self.cap = cv2.VideoCapture(0)
    
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    rate = rospy.Rate(50)  # 5hz
    # record the beginning time
    while not rospy.is_shutdown():
        #print("Sending")
        self.showimg()
        rate.sleep()

        # Recieve the image
        try:
            imP = Image_processes()
            
            #record image frames
            ret, frame = cap.read()
        except CvBridgeError as e:
            print(e)

        # Publish the results
        try: 
            self.robotPosition = imP.runProcessor(frame)
            self.pos_pub.publish(json.dumps({'bench1': self.robotPosition}))
          
        except CvBridgeError as e:
          print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)