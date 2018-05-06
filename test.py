#!/usr/bin/env python3

import random
import cv2
import numpy as np

from process import in_hue_range,hsv2bgr

def random_rgb():
  return [random.randint(0,256) for _ in range(0,3)]

def find_cube(im,mask,base_cnt,hue):

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
  lower = np.array([0,int(msat),0])
  upper = np.array([255,255,255])
  mask = cv2.inRange(sim,lower,upper)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  mask = cv2.erode(mask,kernel)
  mask = cv2.dilate(mask,kernel)

  #cv2.imshow("smask",mask)
  
  tvec = None
  _,contours,_ = cv2.findContours(mask, 1, 2)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  image_points = None
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
      
  cv2.imshow("sim-%d"%(hue),sim)
  return image_points

def search_by_colours(im,hsv,imdbg,hmin,hmax,smin,smax,vmin,vmax):

  h,w,_ = hsv.shape

  # hue selection
  lower = np.array([hmin,smin,vmin])
  upper = np.array([hmax,smax,vmax])
  mask = cv2.inRange(hsv,lower,upper)
  
  # add highlights
  lower = np.array([0,0,256])
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
      smean = (smin+smax)/2
      smean = 50 if smean < 100 else 255
      vmean = (vmin+vmax)/2
      vmean = 50 if vmean < 100 else 255

      color = hsv2bgr(hmean,smean,vmean)
      cv2.drawContours(imdbg, [approx], -1, color, 2)
      for (x,y) in approx[:,0]:
        cv2.circle(imdbg, (x,y), 5, color)

      n = len(approx)
      if n >= 4 and n <=10:
        return find_cube(im,mask,approx,hmean)

  return None

def vision(im,camera_matrix,dist_coeffs):

  h,w,_ = im.shape

  # convert to HSV
  #cv2.imshow("original",im)
  imdbg = im.copy()

  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  hsv = cv2.bilateralFilter(hsv, 5, 50, 50)
 
  COLORS = (
  #(60,120,100,216,21,245), # dark green
  #(93,130,0,255,0,255), # bleu
  (0,30,150,255,0,255), # orange
  #(22,46,118,248,90,255), # yellow
  #(46,93,89,255,0,255), # green
  #(0,255,0,255,0,45), # black
  )
  for (hmin,hmax,smin,smax,vmin,vmax) in COLORS:
    image_points = search_by_colours(im,hsv,imdbg,hmin,hmax,smin,smax,vmin,vmax)
    if image_points is not None:
      a = 0.058
      obj_points = np.array([
        [[0],[0],[0]],
        [[a],[0],[0]],
        [[a],[a],[0]],
        [[0],[a],[0]],
        ])

      # cube position in camera space
      _,rvec,tvec = cv2.solvePnP(obj_points,image_points,camera_matrix,dist_coeffs)
      
      tvcam,rvcam = np.array([[0.003],[-0.003],[0.383]]),np.array([[2.285],[-0.281],[-0.015]])
      rmcam,_ = cv2.Rodrigues(rvcam)

      x,y,z = np.dot(-np.transpose(rmcam),tvcam-tvec)

      cx,cy = [int(c) for c in np.transpose(image_points[0])[0]]
      x,y,z = [int(1000*v) for v in (x,y,z)]
      cv2.putText(imdbg, "%d %d %d"%(x,y,z), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),3)

      #print(tvec)
      #print(rvcam)

  return imdbg

def main_image():
  im = cv2.imread("vlcsnap-2018-04-22-20h20m49s026.png")
  imdbg = vision(im)
  cv2.imshow("cam",imdbg)
  cv2.waitKey()
  cv2.destroyAllWindows()

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

