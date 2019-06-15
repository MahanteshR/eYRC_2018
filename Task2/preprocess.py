# Pre-processes the input image by removing noise and draws the contours for coloured squares in the image

"""
Team Id: HC#3266
Author List: Aman Bhat T, Hemanth Kumar M, Mahantesh R, Nischith B.O.
Filename: preprocess.py
Theme: Homecoming
Functions: preprocess
Global Variables: colour_contours, font, kernel
"""

import numpy as np
import cv2
import copy

colour_contours=0                  # Number of squares with colours
font=cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3,3),np.uint8)   # Create a filter to clean and process the image

def preprocess(image):
    """
    Function name: preprocess
    Input:  image -> original image
    Output: gray -> grayscale image of the input image
            sure_fg -> sure foreground i.e. the colours in the image
            img2 -> Image with the contours drawn for the colours
            colour_contours -> Returns the number of contours for colours
    """
    image=cv2.resize(image, (800,800))    # resizing the image to a convenient size
    img = copy.deepcopy(image)
    img2 = copy.deepcopy(image)    # copies of the original image
    
    # Convert to grayscale and apply suitable threshold to find contours easily
    #-------------------------------------------------------------------------------
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,250,255,0)
    thresh=cv2.bitwise_not(thresh)

    # noise removal
    reduced_noise = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    erosion = cv2.erode(reduced_noise,kernel,iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(erosion,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(erosion,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Adding one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Marking the region of unknown with zero, subtracting from original image to generate only the coloured part separately
    #-----------------------------------------------------------------------------------------------------------------------
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,0,0]
    img3 = img-img2
    img3 = cv2.dilate(img3,kernel,iterations=1)
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img3,10,255,0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours and filter them to display ONLY COLOURED SQUARES
    for i in range(len(contours)):
        ctr=contours[i]
        arc_length = 0.1*cv2.arcLength(ctr,True)
        approx_length = cv2.approxPolyDP(ctr,arc_length,True)
        area = cv2.contourArea(ctr)
        
        if(len(approx_length==4) and hierarchy[0][i][2]==-1 and area > 600):
            global colour_contours
            colour_contours+=1
            img2 = cv2.drawContours(img2,contours, i, (0,0,255), 3)
    
    return gray, img2, sure_fg,colour_contours