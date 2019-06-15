"""
Team Id: HC#3266
Author List: Aman Bhat T, Hemanth Kumar M, Mahantesh R, Nischith B.O.
Filename: imgstrip.py
Theme: Homecoming
Functions: run_module
"""

import cv2
import argparse
import os
import errno

from preprocess import preprocess
from extraction import generate_contours

def run_module():
    """
    Function name: run_module
    Input:  None
    Output: None

    """

    # Create a parser for command line flags
    #---------------------------------------
    ap=argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument("-s", "--save", type=str, required=False, help="Path to save output image")
    ap.add_argument("-c", "--cc", type=str, required=False, help="Path to save found contours")
    args=ap.parse_args()


    cut=False  
    if args.cc:
        cut=True
    
    image_name = args.image
    image = cv2.imread(image_name)

    # Original dimensions of the image
    x_dim = image.shape[0]
    y_dim = image.shape[1]
    
    gray,preprocessed_img,sure_fg,clr_ctr = preprocess(image)
    output_img, contours, contours1 ,num_img, colour_dict, number_dict = generate_contours(gray,preprocessed_img,sure_fg, clr_ctr, cut)
    
    image_copy = cv2.resize(image, (800,800))
    if type(output_img)==(int):
       output_img=image_copy
        
    # Rescaling factors
    xr = int(x_dim/800)         
    yr = int(y_dim/800)
    if xr==0 or yr==0:
        xr=yr=1

    if cut==True:
        # Create a directory with image name. If already existing, use the same
        try:
            os.mkdir(args.cc)           
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # Extracting contours for numbers 
        for i in number_dict.keys():
            x, y, width, height = cv2.boundingRect(contours[int(i)])
            roi = num_img[y:y+height, x:x+width]
            roi = cv2.resize(roi, (width*xr, height*yr))
            cv2.imwrite(os.path.join(args.cc,number_dict[i])+'.png', roi)
        
        # Extracting contours for colours
        for i in colour_dict.keys():
            x, y, width, height = cv2.boundingRect(contours1[int(i)])
            roi = image_copy[y:y+height, x:x+width]
            roi = cv2.resize(roi, (width*xr, height*yr))
            cv2.imwrite(os.path.join(args.cc,colour_dict[i])+'.png', roi)

    # Display the output image in original dimensions
    output_img=cv2.resize(output_img, (x_dim,y_dim))
    cv2.imshow("output",output_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Save the image to specified path if -s flag is given as command line argument
    if args.save:
        cv2.imwrite(args.save,output_img)
    
    return

if __name__ == "__main__":
    run_module()