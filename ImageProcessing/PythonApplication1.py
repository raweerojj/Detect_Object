from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def order_points_old(pts):    
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    #select value the top-left, top-right and bottom-left point
    y = rect[[0, 1], :]
    y1 = np.ptp(y,axis=0)
    x = rect[[0, 3], :]
    x1 = np.ptp(x,axis=0)

    #finding the area with point
    y2 = np.max(y1,axis=0)
    x2 = np.max(x1,axis=0)
    sum = y2*x2

    #call the function will get which the object
    char = object_size(sum)

    # return the ordered coordinates
    return rect,sum,char

def object_size(size):
    if size >= 56000:
        c = 'Phone'
    else:   # grade must be B, C, D or F
        if size >= 5000:
            c = 'Pencil'
        else:  # grade must be C, D or F
            if size >= 1000:
                c = 'Money'
    return c  


#input image, convert to grayscale, and blur it slightly
image = cv2.imread("..\\pic\\test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
edged = cv2.Canny(gray, 50, 100)
cv2.imshow("Canny", edged)
cv2.waitKey(0)

edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("dilate", edged)
cv2.waitKey(0)

edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("dilate_and_erode", edged)
cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts) 

# sort the contours from left-to-right and initialize the bounding box
(cnts, _) = contours.sort_contours(cnts)

# point colors
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))


# loop over the contours
for (i, c) in enumerate(cnts):
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	# rotated bounding box of the contour, then draw the contours
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

    # order the points in the contour 
	rect,sum,char = order_points_old(box)


	print("Found_Object #{}: ".format(i + 1) + char)
	print(rect.astype("int"),"Area =" ,sum)
	print("")
       
	# draw the point with color
	for ((x, y), color) in zip(rect, colors):
		cv2.circle(image, (int(x), int(y)), 5, color, -1)

	# putText num at the top-left corner
	cv2.putText(image,"Object #{}: ".format(i + 1)+ char,
		(int(rect[0][0] - 30), int(rect[0][1] - 40)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

	cv2.imshow("Image", image)
	cv2.waitKey(0)
	
