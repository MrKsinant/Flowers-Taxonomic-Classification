#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################
# PROGRAMMER: Pierre-Antoine Ksinant                         #
# DATE CREATED: 17/03/2020                                   #
# REVISED DATE: -                                            #
# PURPOSE: This file contains the web app workflow functions #
##############################################################

###########################
# Needed packages imports #
###########################

import torch, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


##########################
# Function process_image #
##########################

def process_image(img_path, img_pred_path):
    """
    Orientate, scale, crop and normalize a PIL image for a PyTorch model
    
    Args:
     img_path: input image path for processing
     img_pred_path: processed image path for prediction
     
    Returns:
     img_array: NumPy array
    """
    
    # Open the image corresponding to the given path:
    img = Image.open(img_path)
    
    # Check for EXIF data (not always present):
    exif_orientation_tag = 274
    if (hasattr(img, "_getexif")
            and isinstance(img._getexif(), dict)
            and exif_orientation_tag in img._getexif()):
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]
            
        # Handle EXIF Orientation:
        if orientation == 1:
            # Normal image (nothing to do)
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)
                
    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio:
    img_width, img_height = img.size
    img_ratio = img_height/img_width
    if img_width <= img_height:
        img_s1_width = 256
        img_s1_height = int(img_s1_width*img_ratio)
        img_s1_size = img_s1_width, img_s1_height
        img_s1 = img.resize(img_s1_size)
    else:
        img_s1_height = 256
        img_s1_width = int(img_s1_height/img_ratio)
        img_s1_size = img_s1_width, img_s1_height
        img_s1 = img.resize(img_s1_size)

    # Crop out the center 224x224 portion of the image:
    img_s2_left = (img_s1_width - 224)//2
    img_s2_top = (img_s1_height - 224)//2
    img_s2_width = 224
    img_s2_height = 224
    img_s2_box = (img_s2_left,
                  img_s2_top,
                  img_s2_left + img_s2_width,
                  img_s2_top + img_s2_height)
    img_s2 = img_s1.crop(img_s2_box)
    
    # Save processed image:
    img_s2.save(img_pred_path, quality=95)

    # Modify color channels encoding:
    img_s2_array_0 = np.array(img_s2)/255

    # Normalize color channels encoding:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_s2_array_1 = (img_s2_array_0 - mean)/std

    # Adapt color channels encoding to PyTorch specifications:
    img_array = img_s2_array_1.transpose((2, 0, 1))

    return img_array


##########################
# Function predict_image #
##########################

def predict_image(img_array, gpu=False):
    """
    Predict the class of an image using the trained deep learning model
    
    Args:
     img_array: NumPy array
     gpu: gpu availability
     
    Returns:
     classes: Top classes prediction
     probs: Probabilities associated with top classes prediction
    """

    # Load the trained deep learning model:
    if gpu:
        model = torch.load('models/best-model_CUDA.pth')
    else:
        model = torch.load('models/best-model_CPU.pth')

    # Create a dictionary which maps from category labels to indices:
    with open('data/class_to_idx.json', 'r') as class_to_idx_file:
        class_to_idx = json.load(class_to_idx_file)

    # Create a dictionary which maps from indices to category labels:
    idx_to_class = dict(zip(class_to_idx.values(),
                            class_to_idx.keys()))
    
    # Create a dictionary which maps from category labels to category names:
    with open('data/cat_to_name.json', 'r') as category_names_file:
        cat_to_name = json.load(category_names_file)

    # Image classification by the chosen model on chosen mode:
    model.eval()
    img_tensor = torch.FloatTensor(img_array)
    img_tensor.unsqueeze_(0)
    if gpu:
        img_tensor = img_tensor.to('cuda')
    else:
        img_tensor = img_tensor.to('cpu')
    img_result = model.forward(img_tensor)

    # Get the top 5 most probable classes:
    probs_tensor, classes_tensor = img_result.topk(5)
    if gpu:
        # NumPy doesn't support CUDA:
        probs_tensor = probs_tensor.to('cpu')
    probs_array = probs_tensor.detach().numpy()
    probs = np.exp(probs_array[0])
    probs = probs.tolist()
    probs.reverse()
    if gpu:
        # NumPy doesn't support CUDA:
        classes_tensor = classes_tensor.to('cpu')
    classes_array = classes_tensor.detach().numpy()
    classes_idx = classes_array[0]
    classes = []
    for idx in classes_idx:
        classes.append(cat_to_name[idx_to_class[idx]])
    classes.reverse()

    return classes, probs


#################################
# Function visualize_prediction #
#################################

def visualize_prediction(classes, probs, pred_path):
    """
    Visualize the image prediction performed by the trained deep learning model
    
    Args:
     classes: Top classes prediction
     probs: Probabilities associated with top classes prediction
     pred_path: Image prediction graph path
    """
    
    # Turn interactive plotting off:
    plt.ioff()
    
    # Represent full image prediction:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Configuration of the graph:
    ax.set_title("Top 5 Classification Probabilities For Prediction")
    ax.set_xlim(0., 1.)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mpl.ticker.MaxNLocator(11))
    ax.grid(color="grey", which="major", axis='x', linestyle='solid', linewidth=0.25)
    ax.grid(color="grey", which="minor", axis='x', linestyle='solid', linewidth=0.25)
    ax.barh(np.arange(5), probs)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(classes)
    
    # Save the image prediction plot:
    fig.savefig(pred_path, bbox_inches='tight')