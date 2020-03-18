#!/usr/bin/env python3
# -*- coding: utf-8 -*-


########################################
# PROGRAMMER: Pierre-Antoine Ksinant   #
# DATE CREATED: 18/03/2020             #
# REVISED DATE: -                      #
# PURPOSE: Streamlit web app in action #
########################################


###########################
# Needed packages imports #
###########################

import os, torch
import streamlit as st
from PIL import Image
from app_workflow import process_image, predict_image, visualize_prediction


######################
# Constant variables #
######################

INFERENCE_FOLDER = 'inference/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']
VGG_LOGO = 'animation/vgg_logo.png'


#################################
# Specific code for the web app #
#################################

# Display title:
st.title('Flowers Taxonomic Classification')

# Display a zone for uploading a picture:
uploaded_file = st.file_uploader('Upload a picture representing a flower to identify it:',
                                 type=ALLOWED_EXTENSIONS)

# Display a sidebar zone:
st.sidebar.markdown('Made (with love) thanks to:')
st.sidebar.image(VGG_LOGO, use_column_width=True)
st.sidebar.markdown('**Visual Geometry Group**')
st.sidebar.markdown('Department of Engineering Science, University of Oxford.')

# Trained deep learning model in action:
if uploaded_file is not None:
    
    # Take in charge uploaded file:
    st.markdown('---')
    image = Image.open(uploaded_file)
    filename = 'image.jpg'
    filepath = os.path.join(INFERENCE_FOLDER, filename)
    image.save(filepath, quality=95)
    st.markdown('The picture has been **correctly** uploaded and saved!')
    st.image(image, use_column_width=True)
    
    # Perform taxonomic classification on uploaded image:
    st.markdown('---')
    inputname = 'input.jpg'
    inputpath = os.path.join(INFERENCE_FOLDER, inputname)
    img_array = process_image(filepath, inputpath)
    classes, probs = predict_image(img_array, gpu=torch.cuda.is_available())
    outputname = 'output.jpg'
    outputpath = os.path.join(INFERENCE_FOLDER, outputname)
    visualize_prediction(classes, probs, outputpath)
    st.markdown('Check uploaded picture\'s flower taxonomic classification **results**!')
    output_img = Image.open(outputpath)
    st.image(output_img, use_column_width=True)
    st.markdown('---')
    st.markdown('Uploaded picture\'s **flower specie prediction**:')
    st.write(classes[4])