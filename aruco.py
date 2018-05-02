#!/usr/bin/env python3

import cv2
import numpy as np

def main():

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

  aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

  cal_params = cv2.FileStorage("dell_cam.xml", cv2.FILE_STORAGE_READ)
  camera_matrix = cal_params.getNode("cameraMatrix").mat()
  dist_coeffs = cal_params.getNode("dist_coeffs").mat()

  print(camera_matrix,dist_coeffs)
  aruco_params = cv2.aruco.DetectorParameters_create()

  while True:

    r,im = cap.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    corners,ids,rejected = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=aruco_params)

    if ids is not None:
      
      marker_length = 0.062
      rvec,tvec,_ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

      rvec = rvec[0]
      tvec = tvec[0]
      cv2.aruco.drawDetectedMarkers(im, corners, ids)
      cv2.aruco.drawAxis(im, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

      x,y,z = tvec[0,:]
      x = int(100*x)
      y = int(100*y)
      z = int(100*z)

      cv2.putText(im, "X=%d Y=%d Z=%d"%(x,y,z), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)

    cv2.imshow("in",im)
    key = cv2.waitKey(1)
    if key == 27:
      break

  cv2.destroyAllWindows()
  cv2.VideoCapture(0).release()


if __name__ == '__main__':
  main()


