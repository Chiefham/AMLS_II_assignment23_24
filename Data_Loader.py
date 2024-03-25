import csv
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split


def Data_Loader(data_path,label_path,width=800,height=600):
    # 由于数据过大，因此返回值为两个generator

    images = []
    folder = os.listdir(data_path)

    for file in folder:
        image_path = os.path.join(data_path, file)
        image = cv2.imread(image_path)
        image_array = cv2.resize(image,(width,height))
        image_array = image_array/255.0
        images.append(image_array)
        images = np.array(images)

    with open(label_path, 'r') as lp:
        labels = list(csv.reader(lp))

    # shuffle
    images,labels = shuffle(images,labels,random_state=7777)

    # split
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=7)

    # one-hot code
    y_train = to_categorical(y_train,num_classes=5)
    y_test = to_categorical(y_test, num_classes=5)



    return x_train, x_test, y_train, y_test









