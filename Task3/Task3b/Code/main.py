"""
Team Id: HC#3266
Author List:  Hemanth Kumar K L, Mahantesh R, Aman Bhat T, Nischith B.O.
Filename: main.py
Theme: Homecoming
Functions: run_module , contour , predict
Global variables: class_names 
"""


import cv2
import numpy as np
import torch
import os
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

class_names=[]

def run_module():
    """
    Function name: run_module.
    Calls respective model based on command line flags 
    Input:  None
    Output: None
    """

    # Create a parser for command line flags
    #---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-s", "--save", type=str, required=False)
    ap.add_argument("--amod", type=str, required=False,default="HC#3266animal_model.pth")
    ap.add_argument("--hmod", type=str, required=False,default="HC#3266habitat_model.pth")
    args = ap.parse_args()

    # Checking for animal/habitat model path
    #-------------------------------------------
    if args.image:
        animod_path=args.amod
        habmod_path=args.hmod
    img=contour(args.image,animod_path,habmod_path)
    if args.save:
        cv2.imwrite(args.save,img)

def predict(model,image):
    """
    Function name : predict
    Inputs parameters : 
            model -> trained model
            image -> image to be predicted
    Output : return the predicted image
        
    """
    global class_names

    model.eval()        # Update initial weights of the model
    # pre-processing the image
    # ------------------------
    img_loader = transforms.Compose([transforms.Resize(247),transforms.CenterCrop(224), transforms.ToTensor()])
    pil_image = img_loader(image).float()
    np_image = np.array(pil_image) 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    image = torch.FloatTensor(np_image)
    #image = Variable(image)
    image = image.unsqueeze(0).float()
    
    outputs = model(image)              # pre-processed image is sent to model for prediction.
    _, preds = torch.max(outputs, 1)    # Returns the most likely class name for the image
    pred=class_names[preds[0]]
    return pred

def contour(input_img,animod_path,habmod_path):
    """
    Function name : contour
    Input parameters : 
            input_img -> the input image
            animod_path -> animal model path  (optional)
            habmod_path -> habitat model path  (optional)
    Output : image with contours
    This function returns an image with contours around objects.
    """
    
    output_dict={} 
    global class_names

    img = cv2.pyrUp(cv2.imread(input_img,cv2.IMREAD_UNCHANGED))     # increasing the resolution of the input image
    
    # convert to grayscale, apply threshold and find contours.
    # --------------------------------------------------------
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,250,255,0)
    thresh=cv2.bitwise_not(thresh)
    image,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ret1, thresh1 = cv2.threshold(gray,220,255,0)
    thresh1=cv2.bitwise_not(thresh1)
    _,contours1,hierarchy1=cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    j=animal_loc=habitat_grid_count=animal_grid_count=0
    position=50
    column=row=0
    err=0       # flag to check presence of animal model
    err1=0      # flag to check presence of habitat model
    
    # To check validity of animal model path
    if os.path.exists(animod_path):
        model_conv1=torch.load(animod_path,map_location=lambda storage,loc:storage)
        class_names=model_conv1['class_names']
        model_conv2=model_conv1['model']
    else:
        err=1

    # Naming the contours that contain animals
    # ----------------------------------------
    for i in range(len(contours1)):
        cnt=contours1[i] 
        if hierarchy1[0][i][3]==-1: 
            animal_grid_count=animal_grid_count+1
            if hierarchy1[0][i+1][0] and  hierarchy1[0][i+1][1] and  hierarchy1[0][i+1][2] != -1:  
                x,y,w,h=cv2.boundingRect(cnt)
                roi=img[y:y+h,x:x+w]
                roi=roi[:,:,::-1]       # Convert BGR to RGB
                roi=Image.fromarray(roi,'RGB')
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),25)     # Draw the contour
                font=cv2.FONT_HERSHEY_SIMPLEX
                x=int(x+w/2)
                k=(x-60,y-30)
                if animal_grid_count<7:
                    symbol=ord('F')-animal_grid_count+1
                    animal_loc=chr(symbol)+'1'
                    cv2.putText(img,animal_loc,k,font,4,(255,0,0),10,cv2.LINE_AA)
                elif animal_grid_count in [7,9,11,13]:
                    symbol=ord('F')
                    l=int(animal_grid_count/2)-1
                    animal_loc=chr(symbol)+str(l)
                    cv2.putText(img,animal_loc,k,font,4,(255,0,0),10,cv2.LINE_AA)
                elif animal_grid_count>15 and animal_grid_count<22:
                    symbol=ord('F')-animal_grid_count+16
                    animal_loc=chr(symbol)+'6'
                    cv2.putText(img,animal_loc,k,font,4,(255,0,0),10,cv2.LINE_AA)
                else:
                    symbol=ord('A')
                    l=int(animal_grid_count/2)-2
                    animal_loc=chr(symbol)+str(l)
                    cv2.putText(img,animal_loc,k,font,4,(255,0,0),10,cv2.LINE_AA)

                # Predict the name of the animal in the contours and store them in the dictionary
                # -------------------------------------------------------------------------------
                if err==0:
                    pred=predict(model_conv2,roi)       
                    output_dict[animal_loc]=pred       

    for i in range(len(contours)):
        if hierarchy[0][i][3]==-1: 
            j=j+1
            if j==15:
                k=i

    # Check the validity of the habitat model path
    # --------------------------------------------            
    if os.path.exists(habmod_path):
        model_conv1=torch.load(habmod_path,map_location=lambda storage,loc:storage)
        class_names=model_conv1['class_names']
        model_conv2=model_conv1['model'] 
    else:
        err1=1                   

    # Naming the contours that contain habitats
    # -----------------------------------------
    for i in range(len(contours)):
        cnt=contours[i]
        if hierarchy[0][i][3]==k:   # Find the outermost contours for each habitat
            m=i+1
            n1=0
            habitat_loc=habitat_grid_count+1
            habitat_loc=56-(habitat_loc+position-(10*row))
            column=column+1
            if(column==5):
                row+=1
                column=0
            perimeter1 = cv2.arcLength(cnt,True)
            while hierarchy[0][m][3]==i and n1==0:
                cnt=contours[m]
                perimeter = cv2.arcLength(cnt,True)
                if perimeter > (perimeter1/1.5):
                    x,y,w,h=cv2.boundingRect(cnt)
                    roi=img[y:y+h,x:x+w]
                    roi = roi[:, :, ::-1]       # Convert BGR to RGB
                    roi=Image.fromarray(roi,'RGB')
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),25)     # Draw the contour
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    x=int(x+w/3)
                    y=int(y+h/2)
                    k1=(x-10,y+5)
                    cv2.putText(img,str(habitat_loc),k1,font,4,(0,0,255),25,cv2.LINE_AA)

                    # Predict the names of the habitats in the contours and store them in the dictionary
                    # ----------------------------------------------------------------------------------
                    if err1==0:
                        pred=predict(model_conv2,roi)
                        output_dict[str(habitat_loc)]=pred

                    n1=1
                m=m+1 
            habitat_grid_count=habitat_grid_count+1    
    img = cv2.pyrDown(img,cv2.IMREAD_UNCHANGED)     # Bring back original resolution of input image
    img1=cv2.resize(img,(700,700))

    # Print the positions and predicted names of animals and habitats in the form of a dictionary
    print("\n--------DICTIONARY OUTPUT--------\n")
    if err==0 and err1==1:
        print("HABITAT MODEL NOT FOUND\nPREDICTED ONLY ANIMALS\n")
    if err==1 and err1==0:
        print("ANIMAL MODEL NOT FOUND\nPREDICTED ONLY HABITATS\n")
    if err==1 and err1==1:
        print("NO MODELS FOUND TO PREDICT RECOGNISED OBJECTS\n")
    print("{ location   :\t animals/habitats")
    print("  --------    \t ----------------")
    for i,j in output_dict.items():
        print("  %2s"%format(i),"\t     :   ",j)
    print('}')
    cv2.imshow("OBJECTS IDENTIFIED",img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img
 

if __name__ == "__main__":
    run_module()

