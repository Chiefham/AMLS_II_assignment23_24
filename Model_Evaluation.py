import os
from keras.models import load_model
import cv2
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from keras.utils import to_categorical


def Model_Evaluation(test_generator, model_path, test_label_path):
    # STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    test_generator.reset()

    # load model
    model = load_model(model_path)
    pred = model.predict_generator(test_generator, verbose=1)
    y_pred = np.argmax(pred, axis=1)
    y_true = pd.read_csv(test_label_path)
    y_true = y_true['label'].values
    report = classification_report(y_true, y_pred, digits=6)
    print(report)
