import pandas as pd
from Models import Models
from Data_Loader import TrainGen
from train_test_split import train_test_split
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D,BatchNormalization, Activation
import glob
from Model_Evaluation import Model_Evaluation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras

import os


def main():
    # global setting
    batch_size = 8
    target_size_dim = 300
    epochs=10

    # load data
    train_label_path = '../cassava-leaf-disease-classification/train.csv'
    train_img_path = '../cassava-leaf-disease-classification/train_images/'
    df = pd.read_csv(train_label_path)
    df['path'] = train_img_path + df['image_id']
    df['label'] = df['label'].astype('str')
    X_train, X_valid = train_test_split(df, test_size=0.1, random_state=42,
                                        shuffle=True)
    train_generator = TrainGen(X_train, x_col='path', y_col='label',
                               batch_size=batch_size,
                               target_size_dim=target_size_dim)
    val_generator = TrainGen(X_valid, x_col='path', y_col='label',
                             batch_size=batch_size * 2,
                             target_size_dim=target_size_dim)

    # # train and save model
    MODEL = Models(target_size_dim,train_generator,val_generator,epochs)

    # MODEL.EfficientNetB3()
    # MODEL.VGG19()
    # MODEL.ResNet101V2()

    # model evaluation
    efficientnetb3_path = './EfficientNetB3_WB.model'
    vgg19_path='./VGG19.model'
    resnet101v2 = './ResNet101V2.model'


    Model_Evaluation(efficientnetb3_path, val_generator)




if __name__ == "__main__":
    main()
