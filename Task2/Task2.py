#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import cv2
import copy

colour_contours=0
number_contours=0
font=cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3,3),np.uint8)
d=dict()
d2=dict()

def preprocess(image):
    img = copy.deepcopy(image)
    img2 = copy.deepcopy(image)

    img = cv2.resize(img, (800,800))
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

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,0,0]
    img2 = cv2.resize(img2, (800,800))
    img3 = img-img2
    img3 = cv2.dilate(img3,kernel,iterations=1)
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img3,10,255,0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        ctr=contours[i]
        arc_length = 0.1*cv2.arcLength(ctr,True)
        approx_length = cv2.approxPolyDP(ctr,arc_length,True)
        area = cv2.contourArea(ctr)
        
        if(len(approx_length==4) and hierarchy[0][i][2]==-1 and area > 600):
            global colour_contours
            colour_contours+=1
            img2 = cv2.drawContours(img2,contours, i, (0,0,255), 3)

    return gray, img2, sure_fg


def generate_contours(gray,img2,image, sure_fg, cut):
    img4 = copy.deepcopy(image)
    img4 = cv2.resize(img4, (800,800))
    _, thresh = cv2.threshold(gray,50,255,0)

    thresh = cv2.erode(thresh,kernel,iterations=1)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.bitwise_or(thresh,sure_fg)
    thresh = cv2.bitwise_not(thresh)

    _,contours1,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    colour_grid_count=row=column=number_grid_count=colour_position=0
    position=50

    for i in range(len(contours1)):
        ctr=contours1[i-1]
        perimeter = cv2.arcLength(ctr,True)
        if perimeter<414 and perimeter>399:
            arc_length = 0.1*cv2.arcLength(ctr,True)
            approx_length = cv2.approxPolyDP(ctr,arc_length,True)
            colour_position=colour_grid_count+1
            colour_position=56-(colour_position+position-(10*row))
            column=column+1
            if(column==5):
                row+=1
                column=0
            if(len(approx_length)==4):
                if(colour_grid_count<25 and hierarchy[0][i-1][2]!=-1):
                    global number_contours
                    number_contours+=1
                    cv2.drawContours(img4,contours1,i, (0,0,255), 3)
                    ctr=contours1[i]
                    x,y,w,h=cv2.boundingRect(ctr)
                    x=int(x+w/2)
                    y=int(y+h/2)
                    grid_coordinates=(x-10,y+5)
                    cv2.putText(img2,str(colour_position),grid_coordinates,font,0.5,(0,255,0),2,cv2.LINE_AA)
                    global d
                    d[str(i)]=str(colour_position)
            colour_grid_count+=1
            

    _,thresh = cv2.threshold(gray,127,255,0)
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        ctr=contours[i-1]
        perimeter = cv2.arcLength(ctr,True)
        if perimeter>186 and perimeter<196:
            number_grid_count+=1
            if(number_grid_count<21):
                ctr=contours[i]          
                if((i+1)<len(contours)):
                    ctr2=contours[i+1]
                    correlation2=cv2.matchShapes(ctr,ctr2,1,0.0) 
                    correlation1=cv2.matchShapes(ctr,ctr,1,0.0)
                else:
                    correlation2=0
                    correlation1=0

                if(correlation2>0.02 or correlation1>0.02):
                    cv2.drawContours(img2,contours,i-1, (0,0,255), 3)
                    ctr=contours[i-1]
                    l=list(ctr[ctr[:,:,1].argmin()][0])
                    l[0]=l[0]+15
                    l[1]=l[1]-10
                    number_position=(l[0],l[1])
                    if number_grid_count<7:
                        symbol=ord('F')-number_grid_count+1
                        name=chr(symbol)+'1'
                        cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)
                    elif number_grid_count in [7,9,11,13,15]:
                        symbol=ord('F')
                        l=int(number_grid_count/2)-1
                        name=chr(symbol)+str(l)
                        cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)
                    elif number_grid_count>15 and number_grid_count<21:
                        symbol=ord('F')-number_grid_count+15
                        name=chr(symbol)+'6'
                        cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)
                    else:
                        symbol=ord('A')
                        l=int(number_grid_count/2)-2
                        name=chr(symbol)+str(l)
                        cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)
                    if cut==True:
                        global d2
                        d2[str(i-1)]=name
    if(colour_contours==number_contours):
        imshow=img2
    else:
        imshow=img4
    return imshow,contours,contours1,img4

if __name__ == "__main__":
    args = len(sys.argv)
    cut=False
    if args==4:
        if str(sys.argv[2])=="-s":
            cv2.imwrite(sys.argv[3], imshow)
            print("done ")
        elif str(sys.argv[2])=="-c":
            print("elif")
            cut = True
    image_name = str(sys.argv[1])
    image = cv2.imread(image_name)
    x_dim = image.shape[0]
    y_dim = image.shape[1]
    gray,preprocessed_img,sure_fg = preprocess(image)
    imshow, contours, contours1, img4 = generate_contours(gray,preprocessed_img,image,sure_fg, cut)
    
    xr = int(x_dim/800)
    yr = int(y_dim/800)
    if cut==True:
        print("if")
        for i in d2.keys():
            x, y, width, height = cv2.boundingRect(contours[int(i)])
            roi = imshow[y:y+height, x:x+width]
            roi = cv2.resize(roi, (width*xr, height*yr))
            cv2.imwrite(str(sys.argv[3])+d2[i]+'.png', roi)

        for i in d.keys():
            x, y, width, height = cv2.boundingRect(contours1[int(i)])
            roi = img4[y:y+height, x:x+width]
            roi = cv2.resize(roi, (width*xr, height*yr))
            cv2.imwrite(str(sys.argv[3])+d[i]+'.png', roi)

    imshow=cv2.resize(imshow, (x_dim,y_dim))
    cv2.imshow("output",imshow)
    cv2.waitKey()
    cv2.destroyAllWindows()