import os
from keras.models import load_model
import cv2
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from keras.utils import to_categorical




def Model_Evaluation(test_path,test_label_path,model_path,height,width):

    model = load_model(model_path)
    test_images = []
    folder = os.listdir(test_path)

    for file in folder:
        image_path = os.path.join(test_path, file)
        image = cv2.imread(image_path)
        image_array = cv2.resize(image, (width, height))
        image_array = image_array / 255.0
        test_images.append(image_array)

    y_test = pd.read_csv(test_label_path)
    y_test = y_test['label'].values
    y_test = to_categorical(y_test,num_classes=5)

    y_pred = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    report = classification_report(y_true_classes, y_pred_classes, digits=6)
    print(report)