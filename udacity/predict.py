#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
# PROGRAMMER: Pierre-Antoine Ksinant                                    #
# DATE CREATED: 14/07/2018                                              #
# REVISED DATE: -                                                       #
# PURPOSE: Predict flower name from an image along with the probability #
#          of that name                                                 #
#                                                                       #
# Expected call with <> indicating expected user input:                 #
#      python predict.py </path/to/image> <checkpoint> --top_k <k>      #
#             --category_names <cat.json> --gpu                         #
#                                                                       #
# Example call:                                                         #
#      python predict.py 'marigold.jpg' model.pth                       #
#             --top_k 5 --category_names cat_to_name.json --gpu         #
#########################################################################

###########################
# Needed packages imports #
###########################

import argparse, torch, json, sys
import numpy as np
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image

#########################
# Main program function #
#########################

def main():

    # Creates & retrieves command line arguments:
    in_arg = get_input_args()

    # Makes the prediction:
    probs, classes = predict_image(in_arg['input'], in_arg['checkpoint'],
                                   in_arg['top_k'], in_arg['category_names'],
                                   in_arg['gpu'])

    # Prints the results:
    print_results(in_arg['input'], in_arg['checkpoint'],
                  probs, classes,
                  in_arg['top_k'], in_arg['category_names'],
                  in_arg['gpu'])

###########################
# Function get_input_args #
###########################

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as a dictionary.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     vars(parse_args()) -dictionary structure that stores the command line arguments object
    """

    # Creates parse:
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')

    # Creates 5 command line arguments:
    parser.add_argument('input', type=str,
                        help='path to image')
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint classifier model')
    parser.add_argument('--top_k', type=int, default=5,
                        help='return top k most likely classes (default 5)')
    parser.add_argument('--category_names', type=str, default=None,
                        help='use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='use GPU for inference')

    # Returns parsed argument dictionary:
    return vars(parser.parse_args())

##########################
# Function process_image #
##########################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an NumPy array
    '''

    # Open the image corresponding to the given path:
    try:
        im = Image.open(image)
    except:
        print("The file '{}' can't be found.".format(image))
        sys.exit(1)

    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio:
    im_width, im_height = im.size
    im_ratio = im_height/im_width
    if im_width <= im_height:
        im_s1_width = 256
        im_s1_height = int(im_s1_width*im_ratio)
        im_s1_size = im_s1_width, im_s1_height
        im_s1 = im.resize(im_s1_size)
    else:
        im_s1_height = 256
        im_s1_width = int(im_s1_height/im_ratio)
        im_s1_size = im_s1_width, im_s1_height
        im_s1 = im.resize(im_s1_size)

    # Crop out the center 224x224 portion of the image:
    im_s2_left = (im_s1_width - 224)//2
    im_s2_top = (im_s1_height - 224)//2
    im_s2_width = 224
    im_s2_height = 224
    im_s2_box = (im_s2_left,
                im_s2_top,
                im_s2_left + im_s2_width,
                im_s2_top + im_s2_height)
    im_s2 = im_s1.crop(im_s2_box)

    # Modify color channels encoding:
    im_s2_array_0 = np.array(im_s2)/255

    # Normalize color channels encoding:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im_s2_array_1 = (im_s2_array_0 - mean)/std

    # Adapt color channels encoding to PyTorch specifications:
    im_array = im_s2_array_1.transpose((2, 0, 1))

    # Return NumPy array:
    return im_array

##########################
# Function predict_image #
##########################

def predict_image(image_path, chosen_model, top_k, category_names=None, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Load the chosen model's checkpoint:
    try:
        checkpoint = torch.load(chosen_model)
    except:
        print("The classifier model '{}' can't be found.".format(chosen_model))
        sys.exit(1)

    # Rebuild the model:
    if checkpoint['arch'] == 'densenet161':
        model = models.densenet161()
        model.classifier = nn.Sequential(OrderedDict([('input', nn.Linear(2208, checkpoint['hidden_units'])),
                                                      ('drop1', nn.Dropout(p=0.2)),
                                                      ('act1', nn.ReLU()),
                                                      ('hl1', nn.Linear(checkpoint['hidden_units'], 102)),
                                                      ('output', nn.LogSoftmax(dim=1))]))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.vgg16()
        model.classifier = nn.Sequential(OrderedDict([('input', nn.Linear(25088, checkpoint['hidden_units'])),
                                                      ('drop1', nn.Dropout(p=0.2)),
                                                      ('act1', nn.ReLU()),
                                                      ('hl1', nn.Linear(checkpoint['hidden_units'], 102)),
                                                      ('output', nn.LogSoftmax(dim=1))]))
        model.load_state_dict(checkpoint['state_dict'])

    # Make sure the model matches the chosen mode:
    if gpu:
        model.to('cuda')
    else:
        model.to('cpu')

    # Create a dictionary which maps from category labels to indices:
    with open('../data/class_to_idx.json', 'r') as class_to_idx_file:
        class_to_idx = json.load(class_to_idx_file)

    # Create a dictionary which maps from indices to category labels:
    idx_to_class = dict(zip(class_to_idx.values(),
                            class_to_idx.keys()))

    # Make sure to use a category names file if chosen:
    if category_names != None:
        try:
            with open(category_names, 'r') as category_names_file:
                cat_to_name = json.load(category_names_file)
        except:
            print("The file '{}' can't be found.".format(category_names))
            sys.exit(1)

    # Preprocessing of the image:
    image_array = process_image(image_path)

    # Image classification by the chosen model on chosen mode:
    model.eval()
    image_tensor = torch.FloatTensor(image_array)
    image_tensor.unsqueeze_(0)
    if gpu:
        image_tensor = image_tensor.to('cuda')
    else:
        image_tensor = image_tensor.to('cpu')
    image_result = model.forward(image_tensor)

    # Get the top 'k' most probable classes:
    probs_tensor, classes_tensor = image_result.topk(top_k)
    if gpu:
        # NumPy doesn't support CUDA:
        probs_tensor = probs_tensor.to('cpu')
    probs_array = probs_tensor.detach().numpy()
    probs = np.exp(probs_array[0])
    probs = probs.tolist()
    if gpu:
        # NumPy doesn't support CUDA:
        classes_tensor = classes_tensor.to('cpu')
    classes_array = classes_tensor.detach().numpy()
    classes_idx = classes_array[0]
    classes = []
    for idx in classes_idx:
        # Make sure to use a category names file if chosen:
        if category_names != None:
            classes.append(cat_to_name[idx_to_class[idx]])
        else:
            classes.append(idx_to_class[idx])

    return probs, classes

##########################
# Function print_results #
##########################

def print_results(image_path, chosen_model, probs, classes, top_k, category_names=None, gpu=False):
    ''' Print the results corresponding to the classification inference.
    '''

    # Chosen paramaters:
    print("*** CHOSEN PARAMETERS")
    print("Image path: {}".format(image_path))
    print("Classifier model: {}".format(chosen_model))
    print("Top k: {}".format(top_k))
    print("Category names: {}".format(category_names))
    if gpu:
        print("Mode: GPU")
    else:
        print("Mode: CPU")

    # Classification categories:
    print("*** CLASSIFICATION CATEGORIES")
    for k in range(top_k):
        print("{}: Flower name... {},".format(k + 1, classes[k]),
              "Probability... {:.2f}%".format(100*probs[k]))

############################################
# Call to main function to run the program #
############################################

if __name__ == "__main__":
    main()
