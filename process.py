import numpy as np
import cv2

NAMED_HUES_RANGES = [
    (0, 42, "ORANGE"),
    (43, 110, "GREEN"),
]

def hsv2bgr(h,s,v):
    hsv = np.uint8([[[h,s,v]]])
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple([int(x) for x in color[0][0]])

def in_hue_range(im, hmin, hmax):
    
    lower = np.array([hmin,150,50])
    upper = np.array([hmax,255,255])
    
    mask = cv2.inRange(im,lower,upper)
    return mask

def biggest_blob_by_hue(im, hrange, offset):
    hmin,hmax,_ = hrange

    # apply a hue mask
    mask = in_hue_range(im,hmin,hmax)

    # erode mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)

    # extract contours
    _,contours,_ = cv2.findContours(mask, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE, offset=offset)
    
    # find biggest contour by area
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return None

    # return biggest contour
    max_idx = np.argmax(areas)
    return contours[max_idx],areas[max_idx],hrange

def biggest_blob(im, dbg, offset, hranges):
    tuples = [ biggest_blob_by_hue(im, hrange, offset) for hrange in hranges ]
    # filter out None elements
    tuples = [ x for x in tuples if x is not None]

    if len(tuples) == 0:
        return None,None,None

    # draw contours
    for contour,_,hrange in tuples:
        hmin,hmax,_ = hrange
        hmean = (hmin+hmax)/2
        bgr = hsv2bgr(hmean,255,255)
        cv2.drawContours(dbg, [contour], -1, bgr, 1)

    # select biggest area between colors
    contours, areas, _ = zip(*tuples)
    idx = np.argmax(areas)

    # return biggest area
    return tuples[idx]

def process_cylinder(im, dbg):
    MINIMUM_AREA = 30000
    # extract bottom of image
    h,w,_ = im.shape
    dy = int(h/2)
    offset = (0,dy)
    bim = im[dy:h,0:w]

    contour, area, hrange = biggest_blob(bim, dbg, offset, NAMED_HUES_RANGES)

    if area is None or area < MINIMUM_AREA:
        color = "???"
    else:
        _,_,color = hrange

    _area = "N/A" if area is None else "%6d"%area
    cv2.putText(dbg, "%s %s"%(_area,color), (10,h-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
   
    return area,color

def process_entry(im, dbg):
    MINIMUM_AREA = 5000
    # extract top of image
    h,w,_ = im.shape
    iy,ih,ix,iw = 0,int(h/2),int(w/4),int(w/2)
    offset = (ix,0)
    tim = im[iy:iy+ih,ix:ix+iw]
    
    # draw region of interest
    cv2.rectangle(dbg, (ix,iy), (ix+iw,iy+ih), (200,200,200), 1)

    contour, area, hrange = biggest_blob(tim, dbg, offset, NAMED_HUES_RANGES)

    h = 0
    if area is None or area < MINIMUM_AREA:
        color = "???"
    else:
        _,_,color = hrange

        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(dbg, (x,y), (x+w,y+h), (200,200,200), 2)

    _area = "N/A" if area is None else "%6d"%area
    cv2.putText(dbg, "%s %s h=%d"%(_area,color,h), (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    
    return area,color,h

def process_image(im):
    # flip image for convinience
    im = cv2.flip(im, 0)
    dbg = im

    # convert image to HSV
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    w,h,_ = im.shape
    # extract top of image (meatgrinder)
    re = process_entry(im,dbg)

    # extract bottom of image (cylinder)
    rc = process_cylinder(im,dbg)

    return dbg,re,rc


