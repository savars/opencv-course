#pylint:disable=no-member

import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#edges
#contours decreases canny edges if you apply it in blured pic
canny = cv.Canny(blur, 125, 175) 
cv.imshow('Canny Edges', canny)

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) # below 125 is set as 0 in treshold function.
# cv.imshow('Thresh', thresh)

# takes edges
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 1) # number of images you want can be chosen here:1
cv.imshow('Contours Drawn', blank) 

#trasholding alone can cause misleading while converting contours. 

cv.waitKey(0)
