#!/usr/bin/env python3

import cv2
import numpy as np

def main():

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

  cal_params = cv2.FileStorage("dell_cam.xml", cv2.FILE_STORAGE_READ)
  camera_matrix = cal_params.getNode("cameraMatrix").mat()
  dist_coeffs = cal_params.getNode("dist_coeffs").mat()

  img = cv2.imread('stylo.png')
  h,w,_ = img.shape

  imotter = cv2.imread("otter.jpg")
  imotter = cv2.resize(imotter, (w,h))

  sift = cv2.xfeatures2d.SIFT_create()

  kpb, desb = sift.detectAndCompute(img, None)

  index_params = dict(algorithm = 0, trees=1)
  search_params = dict(checks=50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)
  while True:

    r,im = cap.read()
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h,w,_ = im.shape
    im = cv2.resize(im,(int(w/2),int(h/2)))
    
    im = cv2.undistort(im, camera_matrix, dist_coeffs)

    h,w,_ = im.shape
    mask = np.zeros((h,w,1),np.uint8)
    mask[:,:,:] = 255

    kp, des = sift.detectAndCompute(im,mask)
    if len(kp) > 0:
      matches = flann.knnMatch(desb, des, k=2)

      good = []
      for m,n in matches:
        if m.distance < 0.7*n.distance:
          good.append(m)

      if len(good) > 10:
        src_pts = np.float32([ kpb[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M,mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h,w,_ = img.shape

        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        #im2 = cv2.polylines(im, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
        
        # create a blank mask
        h,w,_ = imotter.shape
        mask = np.ones((h,w,3), np.uint8)
        mask[:,:,:] = 255

        h,w,_ = im.shape
        mask = cv2.warpPerspective(mask,M,(w,h))
        im2 = cv2.bitwise_or(im,mask)
        
      else:
        matchesMask = None
        im2 = im

      if 0:
        draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = 0)

        print(len(good))
        imo = cv2.drawMatches(img, kpb, im2, kp, good, None, **draw_params)
        cv2.imshow("matches",imo)
      else:
        cv2.imshow("out",im2)

    key = cv2.waitKey(1)
    if key == 27:
      break

  cv2.destroyAllWindows()
  cv2.VideoCapture(0).release()

if __name__ == '__main__':
  main()


