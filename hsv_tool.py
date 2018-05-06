import random
import cv2
import numpy as np
from glob import glob
import itertools

from process import in_hue_range,hsv2bgr

def nothing(x):
  pass


from test import search_by_colours

def main():

  images = itertools.cycle(glob("vlcsnap-2018-04-22-*"))

  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);

  cv2.namedWindow("o")
  cv2.createTrackbar("H-","o",60,255,nothing)
  cv2.createTrackbar("H+","o",120,255,nothing)
  cv2.createTrackbar("S-","o",100,255,nothing)
  cv2.createTrackbar("S+","o",216,255,nothing)
  cv2.createTrackbar("V-","o",21,255,nothing)
  cv2.createTrackbar("V+","o",245,255,nothing)

  while True:
    r,im = cap.read()
    
    cv2.imshow("im",im)
    #im = cv2.imread(next(images))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    #hsv = cv2.bilateralFilter(hsv, 5, 120, 120)

    hmin = cv2.getTrackbarPos('H-','o')
    hmax = cv2.getTrackbarPos('H+','o')
    smin = cv2.getTrackbarPos('S-','o')
    smax = cv2.getTrackbarPos('S+','o')
    vmin = cv2.getTrackbarPos('V-','o')
    vmax = cv2.getTrackbarPos('V+','o')

    output = im.copy()
    mask = search_by_colours(im,hsv,output,hmin,hmax,smin,smax,vmin,vmax)
 
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sz = 640,480
    im = cv2.resize(im, sz)

    mask = cv2.resize(mask, sz)
    output = cv2.resize(output, sz)
    dbg = np.hstack((im,
      cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),
      output))

    cv2.imshow("o",dbg)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
      break

  cv2.VideoCapture(0).release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
