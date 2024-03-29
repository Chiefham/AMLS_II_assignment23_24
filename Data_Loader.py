from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


def Data_Loader(train_path, test_path, label_path, BatchSize=8, width=128,
                height=128,):
    labels = pd.read_csv(label_path)
    labels['label'] = labels['label'].astype(str)

    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                 validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        labels,
        directory=train_path,
        batch_size=BatchSize,
        target_size=(height, width),
        subset='training',
        seed=42,
        x_col='image_id',
        y_col='label',
        class_mode='categorical'
    )

    val_gen = ImageDataGenerator(
        validation_split=0.2
    )

    val_generator = val_gen.flow_from_dataframe(
        labels,
        directory=train_path,
        batch_size=BatchSize,
        target_size=(height, width),
        subset="validation",
        seed=42,
        x_col="image_id",
        y_col="label",
        class_mode="categorical"
    )

    testgen = ImageDataGenerator(rescale=1./255)

    test_generator = testgen.flow_from_dataframe(dataframe=labels,
                                                 directory=test_path,
                                                 x_col='image_id',
                                                 y_col='label',
                                                 batch_size=16, seed=42,
                                                 shuffle=False,
                                                 class_mode=None,
                                                 target_size=(height, width))

    return train_generator, val_generator, test_generator
