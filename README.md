# Flowers Taxonomic Classification Web App

This work takes its roots in a study corresponding to the final validation project of the nanodegree *AI Programming With Python*, schooled by **Udacity, Inc.** (https://eu.udacity.com), a for-profit educational organization offering massive open online courses.

Its goal is to provide a deep learning algorithm that can perform a **taxonomic classification of flowers** based on the dataset capitalized by the Visual Geometry Group of the University of Oxford, consisting of 102 flower categories commonly occuring in the United Kingdom (Each class consists of between 40 and 258 images, further details and the complete dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

This classification task is challenged by some relevant difficulties:
* The images have large scale, pose and light variations;
* There are categories that have large variations within the category;
* There are several very similar categories.

Finally, to put in action the deep learning approach that was mobilized to tackle this problem during the study, a web app has been developed, allowing everyone to enjoy the classification model produced.

In this repository, you will find the following files and folders:
* The file ***Dockerfile***, which can be put in action to build the web app image thanks to Docker paradigm;
* The files ***app_streamlit.py*** and ***utils.py***, which allow to run the dedicated web app developed for the project;
* The folder *udacity*, which contains the files *train.py* (a command line Python script which allows the user to perform a training to generate a consistent and performant deep learning model for a taxonomic classification of flowers), *predict.py* (a command line Python script which allows the user to perform a prediction on a submitted flower image and to determine its top *k* most likely belonging categories, based on a model the user has to provide) and *workspace.py* (specific code to run programs properly inside Udacity working environments);
* The folder *notebook*, which contains the file ***flowers-taxonomic-classification.ipynb***, the Jupyter Notebook corresponding to the detailed and explained study, accompanied by the results obtained;
* The folder *inference*, which contains the various folders and files dedicated to model's predictions, both during the study and during web app action;
* The folder *images*, which contains specific pictures illustrating the Jupyter Notebook used for the study, and, too, the logo of the web app;
* The folder *graphs*, which contains all the training logs and graphs generated during the study;
* The folder *data*, which contains the files *cat_to_name.json* and *class_to_idx.json*, dictionaries linking, respectively, category labels to category names, and indices to category labels, to move smoothly inside the dataset;
* The folder *animation*, which contains a demo of the web app at work.

Due to the generated models weights (approximately 115 Mo for each), the folder *models*, which is used in the study to store the various generated models, has not been provided. This is also not the case for the complete dataset (folders *test*, *train* and *valid*, approximately 365 Mo), which can be found at the location specified at the beginning of this README file.

As an other (and last) remark, it can be noticed that the parameters and the structure of the built neural networks are different in the Jupyter Notebook (see folder *notebook*) from the ones present in the two command line Python scripts (see folder *udacity*): This is due to precise specifications provided by Udacity to write these two command line Python scripts (Enter in a terminal the command `python train.py --help`, or the command `python predict.py --help`, to obtain further details about the way to correctly run these two command line Python scripts).

Finally, to conclude, respectively to the requirements, it can be said that the deep learning models built within this study have been constructed thanks to **PyTorch** (https://pytorch.org), the deep learning framework mainly supported and developed by **Facebook** and its **FAIR** (Facebook AI Research) team (https://research.fb.com/category/facebook-ai-research/), and that the web app has been developed thanks to **Streamlit** (https://www.streamlit.io), a light, simple and pythonic open-source app framework for Machine Learning and Data Science, allowing to create beautiful data apps quickly.

Below, you can find an animation illustrating the web app in action:

![Web App Demo](animation/demo.gif)
