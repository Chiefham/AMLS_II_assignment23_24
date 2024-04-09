# AMLS_II_assignment23_24

This document will explain the purpose of the files
that appear in the GitHub repository
---
Datasets:

Function: Contains the image dataset, 
image labels, and label category explanations.
---
main.py:

Function: main function. The function contains calls to other
modules, from top to bottom, hyperparameter setting, data reading,
model training and model evaluation.

How to use: If you need to train one of the three models,
such as EfficientNetB3, set the “train_efficient” parameter to 1,
otherwise set it to 0. When evaluating the models, if you need
to evaluate a certain model, you can point the path parameter to
the path of the model, and then comment out the other calls to
the model evaluation function. For example, if you only need to
evaluate the EfficientNetB3 model, set "efficientnetb3_path"
to '. /EfficientNetB3.model and comment out
Model_Evaluation(vgg19_path, valid_gen) and
Model_Evaluation(resnet101v2, valid_gen),
keeping only Model_Evaluation(efficientnetb3_path, valid_gen).

---
Data_Loader.py:

Function: Use ImageDataGenerator() to
preprocess the image and use the
ImageDataGenerator.flow_from_directory()
method to generate a stream of data from
a directory to read the image data in batches
to solve the problem of not being able to read
the data into RAM at once.
The ImageDataGenerator.flow_from_directory() method
returns a generator object that provides an iterative
interface that allows image data to be loaded from the
directory in a batch fashion during training,
which can be used to generate batches of image
data and their corresponding labels while training the model.

---
Models.py:

Function: The script consists
of a class and four custom methods.
These methods implement the definition
and training of custom top-level,
EfficientNetB3, VGG19 and ResNet101V2 models.
The class is instantiated in main.py
and its methods are called to train different deep
learning models.

How to use:
In Model() method
the hyperparameters of
the custom top level can be modified.
In the base model's method, the hyperparameter
settings can also be modified for different training
purposes. For example, the EfficientNetB3() method can
be called to train an EfficientNetB3 model and draw and
save its training history, or to train an EfficientNetB3
model with class weights and draw and save its training
history. When commenting out class_weight=self.class_weights_dict,
the EfficientNetB3 model is trained, and vice versa for the
EfficientNetB3 model with weights.

---
hist_plot.py:

Function: It includes the hist_plot
custom method, which serves
to draw the corresponding training
history and save it as a .png picture
based on the given training history and image name.

---
cal_class_weight.py:

Function: Calculate the corresponding class weights based 
on the class distribution of the training data, and the
return value is passed to the self.class_weights_dict in Models.py.

---
*.model:

Function: EfficientNetB3.model, VGG19.model, ResNet101V2.model 
are model files of the corresponding networks, 
EfficientNetB3_WB.model, VGG19_WB.model, 
ResNet101V2_WB.model are model files with class 
weights of the corresponding networks, all of which can be loaded using 
load_model() function.

---
*.pkl:

Function: Same naming rule as for *.model files. 
Each file is the training history of the corresponding network model.

---
*.png:

Function: Same naming rule as for *.model files. 
Each file is a training history image drawn from the 
corresponding .pkl file using the draw function in hist_plot.py

---
*.txt:

Function: The same naming rule as *.model files. Each file is 
used for the evaluation results of the corresponding
network model containing parameters such as accuracy, recall, etc.

---
Data_Exploration.ipynb:

Function: This file is used for data exploration.

---
.gitattribute:

Function: This file is used to keep Git LFS trace records.




















