#!/usr/bin/env python3

import random
import cv2
import numpy as np

from process import in_hue_range,hsv2bgr

def random_rgb():
  return [random.randint(0,256) for _ in range(0,3)]

def find_cube(im,mask,base_cnt):

  x,y,w,h = cv2.boundingRect(base_cnt)

  # extract and mask ROI from full image (and mask)
  sim = im[y:y+h,x:x+w]
  smask = mask[y:y+h,x:x+w]
  sim = cv2.bitwise_and(sim,sim,mask=smask)

  #can = cv2.Canny(sim,100,200)
  #cv2.imshow("_canny",can)

  #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
  #can = cv2.morphologyEx(can, cv2.MORPH_CLOSE, kernel)
  #cv2.imshow("canny",can)

  # compute mean saturation
  _,msat,_,_ = cv2.mean(sim, smask)

  # threshold using mean saturation
  lower = np.array([0,int(1.1*msat),0])
  upper = np.array([255,255,255])
  mask = cv2.inRange(sim,lower,upper)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  mask = cv2.erode(mask,kernel)
  mask = cv2.dilate(mask,kernel)

  cv2.imshow("smask",mask)
  
  tvec = None
  _,contours,_ = cv2.findContours(mask, 1, 2)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  if len(contours) > 0:
    cnt = contours[0]
    approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt,True), True)
    cv2.drawContours(sim, [approx], -1, (255,0,0), 1)

    if len(approx) == 4:
      image_points = []
      for point in approx:
        point += [x,y]
        image_points.append(np.transpose(point))
      image_points = np.array(image_points).astype(np.float)

      a = 0.025
      obj_points = np.array([
        [[0],[0],[0]],
        [[a],[0],[0]],
        [[a],[a],[0]],
        [[0],[a],[0]],
        ])

      cal_params = cv2.FileStorage("dell_cam.xml", cv2.FILE_STORAGE_READ)
      camera_matrix = cal_params.getNode("cameraMatrix").mat()
      dist_coeffs = cal_params.getNode("dist_coeffs").mat()

      _,rvec,tvec = cv2.solvePnP(obj_points,image_points,camera_matrix,dist_coeffs)
      print(tvec)

  cv2.imshow("sim",sim)
  return tvec

def search_by_colours(im,hsv,imdbg,hmin,hmax,smin,smax,vmin,vmax):

  h,w,_ = hsv.shape

  # hue selection
  lower = np.array([hmin,smin,vmin])
  upper = np.array([hmax,smax,vmax])
  mask = cv2.inRange(hsv,lower,upper)
  
  # add highlights
  lower = np.array([0,0,245])
  upper = np.array([256,256,256])
  hmask = cv2.inRange(hsv,lower,upper)
  mask = cv2.bitwise_or(mask,hmask)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
  mask = cv2.erode(mask,kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))
  mask = cv2.dilate(mask,kernel)

  cv2.imshow("mask",mask)
  _,contours,_ = cv2.findContours(mask, 1, 2)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)

    # a polygon of size 6 match a cube projection
    is_convex = cv2.isContourConvex(approx)
    #print(len(approx), cv2.contourArea(approx), is_convex)
    if cv2.contourArea(approx) > 0.005*h*w and is_convex:
      hmean = (hmin+hmax)/2
      color = hsv2bgr(hmean,255,255)
      cv2.drawContours(imdbg, [approx], -1, color, 2)
      for (x,y) in approx[:,0]:
        cv2.circle(imdbg, (x,y), 5, color)

      n = len(approx)
      if n >= 4 and n <=6:
        tvec = find_cube(im,mask,approx)
        if tvec is not None:
          M = cv2.moments(approx)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          x,y,z = [int(100*v) for v in tvec]
          cv2.putText(imdbg, "%d %d %d"%(x,y,z), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)

  return mask

def vision(im):

  h,w,_ = im.shape

  # convert to HSV
  #cv2.imshow("original",im)
  imdbg = im.copy()

  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  hsv = cv2.bilateralFilter(hsv, 5, 50, 50)
 
  COLORS = (
  (60,120,100,216,21,245), # dark green
  #(82,116,150,255,90,200), # bleu
  #(0,20,150,255,100,255), # orange
  #(20,40,150,255,100,255), # yellow
  #(50,70,150,255,100,255), # green
  #(0,255,10,130,0,57), # black
  )
  for (hmin,hmax,smin,smax,vmin,vmax) in COLORS:
    search_by_colours(im,hsv,imdbg,hmin,hmax,smin,smax,vmin,vmax)
    return imdbg

def main_image():
  im = cv2.imread("vlcsnap-2018-04-22-20h20m49s026.png")
  imdbg = vision(im)
  cv2.imshow("cam",imdbg)
  cv2.waitKey()
  cv2.destroyAllWindows()

def main_video():

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

  while True:
    r,im = cap.read()
    imdbg = vision(im)
  
    cv2.imshow("cam",imdbg)

    key = cv2.waitKey(10)
    if key == 27:
      break

  cv2.VideoCapture(0).release()
  cv2.imshow("",imdbg)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main_video()

