import keras
import tensorflow as tf
from keras.layers import MaxPooling2D,Flatten,Dense
from keras.layers import Conv2D,BatchNormalization,Dropout
from sklearn.model_selection import GridSearchCV
from cal_class_weight import cal_class_weight
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.applications.efficientnet import EfficientNetB3
from keras import layers,optimizers







class Models:
    def __init__(self,train_generator,val_generator,test_generator):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator



    def SamCNN(self,LR=0.0001,Epochs=50):

        model = keras.models.Sequential()

        # Conv Layer 1
        model.add(Conv2D(filters=32,kernel_size=(5,5),
                         padding='same',activation=tf.nn.relu,input_shape=(128,128,3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3)))

        # Conv Layer 2
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Conv Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Flattening
        model.add(Flatten())

        # FC
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1025, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        # Output
        model.add(Dense(5,activation='softmax'))

        # Compile

        model.compile(
            loss = 'categorical_crossentropy',
            optimizer=keras.optimizers.Adam(lr=LR),
            metrics=['accuracy'],
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)

        # Fit
        STEP_SIZE_TRAIN = self.train_generator.n//self.train_generator.batch_size
        STEP_SIZE_VALID = self.val_generator.n//self.val_generator.batch_size


        # 使用gridsearchcv调参
        # batch_size = [8, 16]
        # epochs = [10, 50]
        # param_grid = dict(batch_size=batch_size, epochs=epochs)
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

        # 计算并设置class_weight
        train_csv_path = './NewDatasets/train_labels.csv'
        class_weights = cal_class_weight(train_csv_path)
        # class_weights = {str(key): value for key, value in class_weights.items()}

        model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=self.val_generator,
            validation_steps=STEP_SIZE_VALID,
            epochs=Epochs,
            class_weight=class_weights,
            callbacks=[callback],
        )


        # Model Save
        model.save('SamCNN.model')


    def VGG19(self):
        model_vgg19 = VGG19(
            weights="imagenet", include_top=False,
            input_shape=(32, 32, 3)
        )
        for layer in model_vgg19.layers[:-1]:
            layer.trainable = False
        top_model = keras.models.Sequential()
        top_model.add(Flatten(input_shape=model_vgg19.output_shape[1:]))
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(5, activation='softmax'))
        model = Model(
            inputs=model_vgg19.input,
            outputs=top_model(model_vgg19.output)
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(lr=0.001),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

        STEP_SIZE_TRAIN = self.train_generator.n // self.train_generator.batch_size
        STEP_SIZE_VALID = self.val_generator.n // self.val_generator.batch_size

        train_csv_path = './NewDatasets/train_labels.csv'
        class_weights = cal_class_weight(train_csv_path)

        model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=self.val_generator,
            validation_steps=STEP_SIZE_VALID,
            epochs=5,
            class_weight=class_weights,
            callbacks=[callback],
        )

        model.save('VGG19.model')

    def EfficientNetB3(self):

        model = keras.models.Sequential()
        model.add(EfficientNetB3(
            include_top=False,weights='imagenet',input_shape=(128,128,3),
            drop_connect_rate=0.3
        ))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(256,activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(5,activation='softmax'))

        optimizer =optimizers.Adam(learning_rate=1e-4)

        STEP_SIZE_TRAIN = self.train_generator.n // self.train_generator.batch_size
        STEP_SIZE_VALID = self.val_generator.n // self.val_generator.batch_size
        model.compile(
            optimizer=optimizer,loss='categorical_crossentropy',metrics=['categorical_accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=2)

        history = model.fit(self.train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=5,
                            verbose=1,validation_data=self.val_generator,validation_steps=STEP_SIZE_VALID,
                            callbacks=[callback])

        model.save('EfficientNetB3.model')





