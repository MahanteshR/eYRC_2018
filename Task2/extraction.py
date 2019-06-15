"""
Team Id: HC#3266
Author List: Aman Bhat T, Hemanth Kumar M, Mahantesh R, Nischith B.O.
Filename: extraction.py
Theme: Homecoming
Functions: generate_contours
Global Variables: number_contours, font, kernel, colour_dict, number_dict
"""
import numpy as np
import cv2
import copy

number_contours=0
font=cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3,3),np.uint8)
colour_dict=dict()
number_dict=dict()

def generate_contours(gray,img2, sure_fg, colour_contours,cut):
    """
    Function name: generate_contours
    Input:  gray -> grayscale image
            img2 -> Image with the contours drawn for the colours
            sure_fg -> sure foreground generated in preprocess
            colour_contours -> number of contours for colours
            cut -> flag to save contours if -c flag is given
    Output: imshow -> output image with contours drawn
            contours, contours1 -> contours for colours and numbers respectively
            num_img -> image with colour contours numbered
            colour_dict, number_dict -> dictionary containing indices of colour/number contour respectively
    """
    
    # Convert to grayscale, apply threshold to generate empty grid and add sure_fg
    #-----------------------------------------------------------------------------
    _, thresh = cv2.threshold(gray,50,255,0)
    thresh = cv2.erode(thresh,kernel,iterations=1)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.bitwise_or(thresh,sure_fg)
    thresh = cv2.bitwise_not(thresh)

    # finding contours for numbering them
    _,contours1,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # variables to store number of squares in the grid i.e. 25 (colour_grid_count), 
    # row and column values and number of squares that have numbers
    colour_grid_count=row=column=number_grid_count=colour_position=0
    position=50 

    # Iterate through the contours and number the squares containing the colours
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
                if(colour_grid_count<25 and hierarchy[0][i-1][2]!=-1):  # to indicate colour contour
                    global number_contours
                    number_contours+=1
                    ctr=contours1[i]
                    x,y,w,h=cv2.boundingRect(ctr)
                    x=int(x+w/2)
                    y=int(y+h/2)
                    grid_coordinates=(x-10,y+5)
                    img2=cv2.putText(img2,str(colour_position),grid_coordinates,font,0.5,(0,255,0),2,cv2.LINE_AA)
                    global colour_dict
                    colour_dict[str(i)]=str(colour_position)
            colour_grid_count+=1

    num_img = copy.deepcopy(img2)
    num_img = cv2.resize(num_img, (800,800))

    # Finding contours for naming the squares with numbers
    _,thresh = cv2.threshold(gray,127,255,0)
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and label the squares containing numbers appropriately
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
                    if number_grid_count<7:                                 # contours A1 to F1
                        symbol=ord('F')-number_grid_count+1
                        name=chr(symbol)+'1'
                        img2=cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)

                    elif number_grid_count in [7,9,11,13,15]:               # contours F2 to F6
                        symbol=ord('F')
                        l=int(number_grid_count/2)-1
                        name=chr(symbol)+str(l)
                        img2=cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)

                    elif number_grid_count>15 and number_grid_count<21:     # contours A6 to F6     
                        symbol=ord('F')-number_grid_count+15
                        name=chr(symbol)+'6'
                        img2=cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)

                    else:                                                   # contours A2 to A6
                        symbol=ord('A')
                        l=int(number_grid_count/2)-2
                        name=chr(symbol)+str(l)
                        img2=cv2.putText(img2,name,number_position,font,0.6,(0,255,0),2,cv2.LINE_AA)

                    # store the index of each contour of a square containing a number in a dictionary along with its label
                    if cut==True:   
                        global number_dict
                        number_dict[str(i-1)]=name

    if(colour_contours==number_contours):
        imshow=img2
    else:
        imshow=-1

    return imshow,contours,contours1, num_img, colour_dict, number_dict