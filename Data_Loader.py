from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split


def TrainGen(dataframe, target_size_dim, x_col, y_col, batch_size=8):
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        brightness_range=[0.7, 1.5],
        rotation_range=30,
        shear_range=0.2,
        fill_mode='nearest',
        zoom_range=[0.3, 0.6],
    )

    train_generator = train_gen.flow_from_dataframe(
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col,
        class_mode="categorical",
        target_size=(target_size_dim, target_size_dim),
        color_mode='rgb',
        batch_size=batch_size)

    return train_generator


def ValGen(dataframe, target_size_dim, x_col, y_col, batch_size=8):
    val_gen = ImageDataGenerator()

    val_generator = val_gen.flow_from_dataframe(
        dataframe=dataframe,
        x_col=x_col,
        y_col=y_col,
        class_mode="categorical",
        target_size=(target_size_dim, target_size_dim),
        batch_size=batch_size,
        shuffle=False
    )

    return val_generator
