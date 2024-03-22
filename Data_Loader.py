import csv
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np


def Data_Loader(data_path,label_path,width=800,height=600):
    images = []
    folder = os.listdir(data_path)

    for file in folder:
        image_path = os.path.join(data_path, file)
        image = cv2.imread(image_path)
        image_array = cv2.resize(image_array,(width,height))
        image_array = image_array/255.0
        images.append(image_array)

    with open(label_path, 'r') as lp:
        labels = list(csv.reader(lp))

    df = pd.DataFrame(images, columns=labels)

    return df









