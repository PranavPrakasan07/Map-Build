import cv2
import numpy as np
import random
from PIL import Image

#from cv2.ximgproc import guidedFilter

filename = 'output.png'
image = cv2.imread(filename)

#image = cv2.guidedFilter(image, 15, 80, 80,None)
image = cv2.bilateralFilter(image, 15, 80, 80,None)

cv2.imshow('smooth', image)
cv2.imwrite('./results/1-Smooth.png', image)

#cv2.waitKey(0)  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  
cv2.imshow('gray', gray)
cv2.imwrite('./results/2-Gray.png', gray)

#cv2.waitKey(0) 

# Find Canny edges
# edged = cv2.Canny(gray, 30, 200)
thr1=50
thr2=200


kernel = np.ones((5,5 ),np.float32)/49
# gray = cv2.filter2D(gray,-1,kernel)

# kernel = np.ones((3,3), np.uint8)

gray = cv2.dilate(gray, kernel, iterations=3)

cv2.imshow('Edged dilate', gray)
cv2.imwrite('./results/3-Edged Dilate.png', gray)

#cv2.waitKey(0)

gray = cv2.erode(gray, kernel, iterations=1)

cv2.imshow('Edged erode', gray)
cv2.imwrite('./results/4-Edged Erode.png', gray)

#cv2.waitKey(0)

#Check mean

im = Image.open(filename)
im_grey = im.convert('LA') # convert to grayscale
width, height = im.size

total = 0
for i in range(0, width):
    for j in range(0, height):
        total += im_grey.getpixel((i,j))[0]

mean = total / (width * height)
print("Mean: ", mean)

#End check mean

edged = cv2.Canny(gray, thr1, thr2)
cv2.imshow("CannyImg_"+str(thr1) + "_" + str(thr2), edged)
cv2.imwrite("./results/5-Canny Img_"+str(thr1) + "_" + str(thr2) + ".png", edged)

#cv2.waitKey(0)
kernel = np.ones((3,3), np.uint8)


contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


  
print("Number of Contours found = " + str(len(contours)))


cntFound = 0
finalCnt = []
for cnt in contours :
    area = cv2.contourArea(cnt)
    print("area",area)
   
    # Shortlisting the regions based on there area.
    if area > 100: 
        # approx = cv2.approxPolyDP(cnt, 
        #                           0.009 * cv2.arcLength(cnt, True), True)
        
        approx = cv2.approxPolyDP(cnt,0.001 * cv2.arcLength(cnt, True), True)
        
        
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
   
        # Checking if the no. of sides of the selected region is 7.
        # if(len(approx) == 7): 
        r=random.randint(0,255)
        g=random.randint(0,255)
        b=random.randint(0,255)
        cv2.drawContours(image, [approx], -1, (r, g, b), 3)
        cv2.putText(image, str(cntFound), (cX - 20, cY - 20),cv2.QT_FONT_NORMAL, 0.5, (r, g, b), 2)
        
        cntFound = 1 + cntFound
        finalCnt.append(cnt)

print("Total found after area threshold = ", cntFound)  
cv2.imshow('Contours', image)
cv2.imwrite('./results/6-Buildings.png', image)

#cv2.waitKey(0)
cv2.destroyAllWindows()
