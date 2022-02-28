#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import math


class joint_estimation_2:
    def __init__(self):

        self.x_axis = np.array([1, 0, 0])
        self.y_axis = np.array([0, 1, 0])
        self.z_axis = np.array([0, 0, 1])

        # Last known configuration of the system. Initially empty
        # index 0: ja1, index 1: ja3, index2: ja4
        self.last_known_ja = [0,0,0]

        # number of times we have not allowed the angle to change
        # index 0: ja1, index1: ja3, and index 2: ja4
        self.buffer_graph_smoothing = [0, 0, 0]

        # Colour Ranges to be used for thresholding
        # RED
        self.RED_BLOB_HSV_COLOR_RANGE_BELOW = (0,50,50)
        self.RED_BLOB_HSV_COLOR_RANGE_UPPER = (20,255,255)
        # GREEN
        self.GREEN_BLOB_HSV_COLOR_RANGE_BELOW = (50,50,50)
        self.GREEN_BLOB_HSV_COLOR_RANGE_UPPER = (70,255,255)
        # BLUE
        self.BLUE_BLOB_HSV_COLOR_RANGE_UPPER = (130,255,255)
        self.BLUE_BLOB_HSV_COLOR_RANGE_BELOW = (110,50,50)
        # YELLOW
        self.YELLOW_BLOB_HSV_COLOR_RANGE_BELOW = (20,50,50)
        self.YELLOW_BLOB_HSV_COLOR_RANGE_UPPER = (40,255,255)


    # Calculate the conversion from pixel to meter
    def pixel2meter(self,image):
        # Obtain the centre of each coloured blob
        circle1Pos, _ = self.detect_color(self.yz_image, self.xz_image, "yellow")
        circle2Pos, _ = self.detect_color(self.yz_image, self.xz_image, "green")
        # find the distance between two circles
        dist = np.sum((circle1Pos[0:3] - circle2Pos[0:3])**2)
        return self.link_lengths["link_1"] / np.sqrt(dist)
        

      # Returns the position and visibility of a given coloured blob from given two images
    def detect_color(self, image, color):
        color_range_upper = 0
        color_range_below = 0
        if (color == "red"):
            color_range_upper = self.RED_BLOB_HSV_COLOR_RANGE_UPPER
            color_range_below = self.RED_BLOB_HSV_COLOR_RANGE_BELOW
        elif (color == "blue"):
            color_range_upper = self.BLUE_BLOB_HSV_COLOR_RANGE_UPPER
            color_range_below = self.BLUE_BLOB_HSV_COLOR_RANGE_BELOW
        elif(color == "yellow"):
            color_range_upper = self.YELLOW_BLOB_HSV_COLOR_RANGE_UPPER
            color_range_below = self.YELLOW_BLOB_HSV_COLOR_RANGE_BELOW
        elif (color == 'green'):
            color_range_upper = self.GREEN_BLOB_HSV_COLOR_RANGE_UPPER
            color_range_below = self.GREEN_BLOB_HSV_COLOR_RANGE_BELOW

    
        (x, y), small_area_yz = self.find_moments(image, color_range_below, color_range_upper)

        return np.array([x, y])
    

    # Calculates and returns the position and visibility of a coloured blob in a given image based on the given threshold
    # Makes use of hue, saturation, and value in hsv space to calculate required properties.
    # Appropriate values of hue, saturation, and value for different coloured blobs are determined using the 
    # python file hsv_color_finder.py in this package.
    def find_moments(self, image, color_range_below, color_range_upper):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yz_mask = cv2.inRange(hsv_image, color_range_below, color_range_upper)
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        yz_mask = cv2.dilate(yz_mask, kernel, iterations=3)
        # Obtain the moments of the binary image
        M = cv2.moments(yz_mask)
        # Calculate pixel coordinates for the centre of the blob
        m10 = M['m10']
        m00 = M['m00']
        m01 = M['m01']
        # small area determines if this blob is visible
        small_area = False
        if m00 < 100000:
            print("Small moment!")
            small_area = True

        if m00 == 0:
            m00 = 0.000001

        return ((int(m10 / m00), int(m01 / m00)), small_area)
    

    # Draws white circles on the blobs which reftlect the corresponding position of each blob calculated by the algorithm. 
    # Mainly used for debugging
    def draw_circles_on_blobs(self, circle1Pos_img, circle2Pos_img, circle3Pos_img, circle4Pos_img):
        image_with_centers = cv2.circle(self.xz_image, (int(circle1Pos_img[0]), int(circle1Pos_img[2])), 2, (255, 255, 255), cv2.FILLED)
        image_with_centers = cv2.circle(image_with_centers, (int(circle2Pos_img[0]), int(circle2Pos_img[2])), 2, (255, 255, 255), cv2.FILLED)
        image_with_centers = cv2.circle(image_with_centers, (int(circle3Pos_img[0]), int(circle3Pos_img[2])), 2, (255, 255, 255), cv2.FILLED)
        image_with_centers = cv2.circle(image_with_centers, (int(circle4Pos_img[0]), int(circle4Pos_img[2])), 2, (255, 255, 255), cv2.FILLED)

        cv2.imshow('Images with blob centers XZ', cv2.resize(image_with_centers, (400,400)))

        image_with_centers = cv2.circle(self.yz_image, (int(circle1Pos_img[1]), int(circle1Pos_img[3])), 2, (255, 255, 255), cv2.FILLED)
        image_with_centers = cv2.circle(image_with_centers, (int(circle2Pos_img[1]), int(circle2Pos_img[3])), 2, (255, 255, 255), cv2.FILLED)
        image_with_centers = cv2.circle(image_with_centers, (int(circle3Pos_img[1]), int(circle3Pos_img[3])), 2, (255, 255, 255), cv2.FILLED)
        image_with_centers = cv2.circle(image_with_centers, (int(circle4Pos_img[1]), int(circle4Pos_img[3])), 2, (255, 255, 255), cv2.FILLED)

        cv2.imshow('Images with blob centers YZ', cv2.resize(image_with_centers, (400,400)))
    
# call the class
def main(args):
    joint_estimation = joint_estimation_2()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)