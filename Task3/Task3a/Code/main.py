"""
Team Id: HC#3266
Author List:  Hemanth Kumar K L, Mahantesh R, Aman Bhat T, Nischith B.O.
Filename: main.py
Theme: Homecoming
Functions: run_module , imshow , predict
Global variables: class_names 
"""

import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import warnings
warnings.filterwarnings("ignore")

class_names=[]                  # List to store the different class names for animals and habitats

def run_module():
    """
    Function name: run_module.
    Calls respective model based on command line flags 
    Input:  None
    Output: None
    """
    global class_names
    
    # Create a parser for command line flags
    #---------------------------------------
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-a", "--animal", type=str, required=False)
    ap.add_argument("-h", "--habitat", type=str, required=False)
    ap.add_argument("--amod",type=str, default="HC#3266animal_model.pth",required=False)
    ap.add_argument("--hmod",type=str, default="HC#3266habitat_model.pth",required=False)
    args = ap.parse_args()

    # Checking for animal/habitat model path
    #-------------------------------------------
    if args.animal:
        if args.amod:
            model = torch.load(args.amod, map_location=lambda storage, loc: storage)
        else:
            model = torch.load("HC#3266animal_model.pth", map_location=lambda storage, loc: storage)

        class_names = model['class_names']      # Extract class names from model
        trained_model = model['model']          # Extract weights from model and save it to trained model
        predict(trained_model, args.animal)     # Call function to predict the animal

    if args.habitat:
        if args.hmod:
            model = torch.load(args.hmod, map_location=lambda storage, loc: storage)
        else:
            model = torch.load("HC#3266habitat_model.pth", map_location=lambda storage, loc: storage)

        class_names = model['class_names']      # Extract class names from model
        trained_model = model['model']          # Extract weights from model and save it to trained model
        predict(trained_model, args.habitat)    # Call function to predict the habitat



def imshow(inp_img, title=None):
    """
    Function_name -> imshow
    input parameters:
            inp_img -> image to be displayed
            title -> Predicted title by the model
    output: Displays the output image with predicted title

    """
    # Transpose the image to extract mean and standard deviation to add back the clipped parts
    # ----------------------------------------------------------------------------------------

    inp_img = inp_img.numpy().transpose((1, 2, 0))      
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp_img = std * inp_img + mean
    inp_img = np.clip(inp_img, 0, 1)
    plt.imshow(inp_img)

    # Display the prediction as the title
    # -----------------------------------
    if title is not None:   
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated
    

def predict(model,image):
    """
    Function name : predict
    Input parameters :
        model -> trained model
        image -> the image to be predicted
    Output : None
    Generates the output image with predicted title
    
    """
    global class_names
    model.eval()    # Update initial weights of the model
    # pre-processing the image
    # -------------------------
    img_loader = transforms.Compose([ transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    desired_size=247
    img = Image.open(image)
    
    
    # Resize the image to 224*224 to CenterCrop the image, paste it over a black RGB image, and then convert to tensor
    # ----------------------------------------------------------------------------------------------------------------
    display_img=img.resize((224,224),Image.ANTIALIAS)
    blank = Image.new("RGB", (224, 224))        
    blank.paste(display_img, (0, 0))
    displayimage_tensor = img_loader(blank).float()
    display_img = displayimage_tensor.unsqueeze(0).float()


    # Resize the image to desired size, paste it over a black RGB image, and then convert to tensor to send to the model
    # ------------------------------------------------------------------------------------------------------------------
    model_img=img.resize((247,247),Image.ANTIALIAS)
    blank = Image.new("RGB", (desired_size, desired_size))
    blank.paste(model_img, (0,0))
    modelimage_tensor = img_loader(blank).float()
    model_img = modelimage_tensor.unsqueeze(0).float()
    
    outputs = model(model_img)          # sending the pre-processed image to the model for prediction.
    _, preds = torch.max(outputs, 1)    # Returns the most likely class name for the image
    
    ax = plt.subplot(111)               # Select the position for image
    ax.axis('off')
    ax.set_title('predicted: {}'.format(class_names[preds[0]]))     # Append the prediction to the title
    imshow(display_img.cpu().data[0])   # Call function to plot the output image


if __name__ == "__main__":
    run_module()    
