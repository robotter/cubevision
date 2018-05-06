#!/usr/bin/env python3

import random
import cv2
import numpy as np

from process import in_hue_range,hsv2bgr

def vision(im, camera_matrix, dist_coeffs):
  imdbg = im.copy()

  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  #hsv = cv2.bilateralFilter(hsv, 5, 50, 50)

  # orange
  polys = detect_by_hue(hsv,imdbg,0,20)
  # green
  #detect_by_hue(hsv,imdbg,60,80)
  # yellow
  #detect_by_hue(hsv,imdbg,30,50)
  # blue
  #detect_by_hue(hsv,imdbg,100,140)


  a = 0.058
  obj_points = np.array([
      [[-a/2],[-a/2],[0]],
      [[a/2],[-a/2],[0]],
      [[a/2],[a/2],[0]],
      [[-a/2],[a/2],[0]],
      ])

  # cube position in camera space
  if len(polys) > 0:
    for poly in polys:
      
      image_points = []
      for point in poly:
        image_points.append(np.transpose(point))
      image_points = np.array(image_points).astype(float)
      
      if len(image_points) == 4:
        _,rvec,tvec = cv2.solvePnP(obj_points,image_points,camera_matrix,dist_coeffs)
 
        tvcam,rvcam = np.array([[-0.057],[ 0.015],[0.413]]),np.array([[2.040],[0.970],[-0.408]])
        rmcam,_ = cv2.Rodrigues(rvcam)
        
        x,y,z = np.dot(-np.transpose(rmcam), tvcam-tvec)
        x,y,z = [int(1000*v) for v in (x,y,z)]
        cx,cy = [int(c) for c in np.transpose(image_points[0])[0]]
        cv2.putText(imdbg, "%d %d %d"%(x,y,z), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),3)

  return imdbg

def detect_by_hue(hsv,imdbg,hmin, hmax):
  h,w,_ = hsv.shape
  smask = np.zeros((h,w,1),np.uint8)

  dh = 10
  polygons = []
  for v in range(100,250,dh):
    for s in range(100,250,dh):
      lower = np.array([hmin,s,v])
      upper = np.array([hmax,250,250])
      mask = cv2.inRange(hsv,lower,upper)
      
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
      mask = cv2.dilate(mask, kernel)
      mask = cv2.erode(mask, kernel)

      _,contours,_ = cv2.findContours(mask, 1, 2)
      contours = sorted(contours, key=cv2.contourArea, reverse=True)
      for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)

        is_convex = cv2.isContourConvex(approx)
        if cv2.contourArea(approx) > 0.005*h*w:
          n = len(approx)
          p = approx[:,0,:]
          if is_convex and n == 4:
            m = np.zeros((h,w,1),np.uint8)
            cv2.fillPoly(m, [p], (5*dh,))
            smask = cv2.add(smask,m)
          elif not is_convex and n == 6:
            m = np.zeros((h,w,1),np.uint8)
            cv2.fillPoly(m, [p], (5*dh,))
            smask = cv2.add(smask,m)
    
  smask = cv2.erode(smask,kernel)
  smask = cv2.dilate(smask,kernel)
  smask = cv2.inRange(smask, (50,), (255,))

  cv2.imshow("smask",smask)
  _,contours,_ = cv2.findContours(smask, 1, 2)

  polys = []
  for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)
    cv2.drawContours(imdbg, [approx], -1, (255,0,0), 2)
    polys.append(approx)
  
  return polys

def main_video():

  cal_params = cv2.FileStorage("/home/jdam/jevois_640.txt", cv2.FILE_STORAGE_READ)
  camera_matrix = cal_params.getNode("cameraMatrix").mat()
  dist_coeffs = cal_params.getNode("dist_coeffs").mat()

  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);

  while True:
    r,im = cap.read()

    imdbg = vision(im, camera_matrix, dist_coeffs)
    cv2.imshow("",imdbg)
    key = cv2.waitKey(10)
    if key == 27:
      break

  cv2.VideoCapture(0).release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main_video()

