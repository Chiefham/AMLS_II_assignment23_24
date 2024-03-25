from keras.preprocessing.image import ImageDataGenerator
import pandas as pd



def Data_Loader(train_path,test_path,label_path,BatchSize,width=800,height=600,):


    labels = pd.read_csv(label_path)

    datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(dataframe=labels,directory=train_path,x_col='image_id',y_col='label',
                                subset='training',batch_size=BatchSize,seed=77,shuffle=True,
                                class_mode='categorical',target_size=(height,width))
    val_generator = datagen.flow_from_dataframe(dataframe=labels,directory=train_path,x_col='image_id',y_col='label',
                                subset='validation',batch_size=BatchSize,seed=77,shuffle=True,
                                class_mode='categorical',target_size=(height,width))

    testgen = ImageDataGenerator(rescale=1./255)

    test_generator = datagen.flow_from_dataframe(dataframe=labels, directory=test_path, x_col='image_id', y_col='label',
                                                batch_size=BatchSize, seed=77, shuffle=False,
                                                class_mode=None, target_size=(height, width))


    return train_generator,val_generator,test_generator