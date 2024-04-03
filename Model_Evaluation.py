import os
from keras.models import load_model
import cv2
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report


def Model_Evaluation(model_path,val_generator):
    # load model
    my_model = load_model(model_path)

    pred_valid_y = my_model.predict(val_generator, verbose=True)
    pred_valid_y_labels = np.argmax(pred_valid_y, axis=-1)
    valid_labels = val_generator.labels

    print(classification_report(valid_labels, pred_valid_y_labels))