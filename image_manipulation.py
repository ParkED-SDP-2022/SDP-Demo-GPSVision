#!/usr/bin/env python

import cv2
import numpy as np
#----------------------------------------------------------------------------------------------------------

class Image_processes:

    def _init_(self):
        self.calibratred =  False
        return
    
    def runProcessor(self,frame):
        return imageSegmentation(frame)
    
    # Perform image processing
    def __imageSegmentation(self, image):
    
        img = image
        
        print(img.shape) # Print image shape
        cv2.imshow("original", img)
        
        #get original feed dimentions
        height, width = img.shape[:2]

        # Cropping images
        cropped_image1 = img[0        :width/2,   0        :height/2]
        cropped_image2 = img[width/2  :width  ,   0        :height/2]
        cropped_image3 = img[0        :width/2,   height/2 :height  ]
        cropped_image4 = img[width*/2 :width  ,   height/2 :height  ]

        images = [cropped_image1, cropped_image2, cropped_image3, cropped_image4]
        
        
        for i in images:
            
            # Display cropped image
            cv2.imshow("cropped", i)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        
        return self.distortionCorrection(images)
        
    
    def __distortionCorrection(self, images):
        
        #code here to correct lens distortion in 4 images from the camera feed
        if not self.calibratred:
            
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((6*7,3), np.float32)
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world spaceeria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((6*7,3), np.float32)
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            images = glob.glob('*.jpg')

            for fname in images:
                img = cv.imread(fname)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (7,6), None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners)
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, (7,6), corners2, ret)
                    cv.imshow('img', img)
                    cv.waitKey(500)
            cv.destroyAllWindows()
        
        img = cv.imread('left12.jpg')
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMa
        trix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('calibresult.png', dst)

        return self.imageStitch(images)

    
    def __imageStitch(self, images):
    
        #code here to stitch images together from the camera feed
        
        return self.colourSpaceCoordinate(image)
        
    
    def __colourSpaceCoordinate(self, image):

            red_u = (20,20,256)
            red_l = (0,0,100)
            climits = [[red_l,red_u]]
            
            masks = [cv2.inRange(image, climit[0], climit[1]) for climit in climits]
            maskJs = [cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) for mask in masks]
          
            frames = [(image&maskJ) for maskJ in maskJs]
            
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
            
            jThreshes = [cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY) for gray_frame in gray_frames]
            
            jcontours = [cv2.findContours(jthresh[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) for jthresh in jThreshes]
           
            cords = []
            radiuslist = []
            for jcontour in jcontours:
                # print(jcontour)
                try:    
                    Gradius = 0
                    (Gx,Gy),Gradius = cv2.minEnclosingCircle(self.mergeContors(jcontour[0]))
                    radiuslist.append(Gradius)
                    # print(Gradius)
                    if Gradius < 2: #Filter out single pixel showing
                        cords.append([-1,-1])
                    else:
                        cords.append([Gx,Gy])
                                
                except:
                    cords.append([-1,-1])
                    radiuslist.append(0)

            contourDic = {"Red": {'x':cords[3][0],'y':cords[3][1]}}
            
            im_copy = image.copy()
            
            for i in range(len(cords)):
                    cv2.circle(im_copy, (int(cords[i][0]), int(cords[i][1])), 2, (255, 255, 255), -1)
                    cv2.putText(im_copy, list(contourDic.keys())[i], (int(cords[i][0]) - 50, int(cords[i][1]) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.circle(im_copy,(int(cords[i][0]), int(cords[i][1])),int(radiuslist[i]),(0,255,0),1)
                   
            return contourDic, im_copy
        
    def __mergeContors(self, ctrs):
            list_of_pts = []
            for c in ctrs:
                    for e in c:
                            list_of_pts.append(e)
            ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
            ctr = cv2.convexHull(ctr)
            return ctr
    
    def sdpPixelToDegrees(self):
        
