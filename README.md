# Flowers Taxonomic Classification

This study is part of the final project of Nanodegree *AI Programming With Python* schooled by **Udacity, Inc.** (https://eu.udacity.com), a for-profit educational organization offering massive open online courses.

Its goal is to provide a deep learning algorithm that can perform a **taxonomic classification of flowers** based on the dataset capitalized by the Visual Geometry Group of the University of Oxford, consisting of 102 flower categories commonly occuring in the United Kingdom (Each class consists of between 40 and 258 images, further details and the complete dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

This classification is challenged by some relevant difficulties:
* The images have large scale, pose and light variations;
* There are categories that have large variations within the category;
* There are several very similar categories.

In this repository, you will find the following files and folders:
* The file ***flowers-taxonomic-classification.ipynb***, the Jupyter Notebook which contains all the detailed and explained study, accompanied by the results obtained;
* The file ***train.py***, a command line Python script which allows the user to perform a training to generate a consistent and performant deep learning model for a taxonomic classification of flowers;
* The file ***predict.py***, a command line Python script which allows the user to perform a prediction on a submitted flower image and to determine its top *k* most likely belonging categories (this prediction is based on a model the user has to provide);
* The files *cat_to_name.json* and *class_to_idx.json*, which contain dictionaries linking category labels to category names, and indices to category labels, to move smoothly inside the dataset;
* The file *workspace_utils.py*, which contains specific code to run programs properly inside Udacity working environments;
* The folder *assets*, which contains specific pictures illustrating the Jupyter Notebook provided in this repository;
* The folder *graphs*, which contains all the graphs generated during the study;
* The folder *inputs*, which contains the flower images which have been used for prediction during the study;
* The folder *outputs*, which contains the result images after model predictions;
* The folder *training*, which contains the various logs generated during the training phase of the model (logs allowing to plot the graphs present in the folder *graphs*).

Due to the fact of the generated models weights (approximately 115 Mo for each), I did not provide in the repository the folder *models*, which is used in the study to store the various generated models. I also did not provide the folder *flowers* (approximately 365 Mo), which contains the complete dataset and can be found at the location specified at the beginning of this README file.

As an other (and last) remark, it can be noticed that the parameters and the structure of the built neural networks are different in the Jupyter Notebook from the ones present in the two command line Python scripts: This is due to precise specifications provided by Udacity to write these two command line Python scripts (Enter in a terminal the command `python train.py --help` and the command `python predict.py --help` to obtain further details about the way to correctly run these two command line Python scripts).

Finally, to conclude, respectively to the requirements, it can be said that the deep learning models built within this study have been constructed thanks to **PyTorch** (https://pytorch.org), the deep learning framework mainly supported and developed by **Facebook** and its **FAIR** (Facebook AI Research) team (https://research.fb.com/category/facebook-ai-research/).
