import keras
import tensorflow as tf
from keras.layers import MaxPooling2D, Flatten, Dense
from keras.layers import Conv2D, BatchNormalization, Dropout
from sklearn.model_selection import GridSearchCV
from cal_class_weight import cal_class_weight
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.applications.efficientnet import EfficientNetB3
from keras import layers, optimizers
from keras import models, layers, optimizers



class Models:
    def __init__(self, train_generator, val_generator,test_generator, img_height=128, img_width=128):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.img_height = img_height
        self.img_width = img_width

    def EfficientNetB3(self):
        model = models.Sequential()
        model.add(EfficientNetB3(include_top=False, weights='imagenet',
                                 input_shape=(self.img_height,
                                              self.img_width, 3),
                                 drop_connect_rate=0.3))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(5, activation='softmax'))

        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.0001,
            name='categorical_crossentropy'
        )

        optimizer = optimizers.Adam(learning_rate=1e-4)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['categorical_accuracy']
        )

        # model.build((None))

        Steps_per_train = float(self.train_generator.n) / self.train_generator.batch_size
        Steps_per_val = float(self.val_generator.n) / self.val_generator.batch_size

        rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                      factor=0.2,
                                                      mode="min",
                                                      min_lr=1e-6,
                                                      patience=2,
                                                      verbose=1)

        estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 mode="min",
                                                 patience=3,
                                                 verbose=1,
                                                 restore_best_weights=True)

        model.fit(
            self.train_generator,
            steps_per_epoch=int(Steps_per_train),
            epochs=5,
            verbose=1,
            validation_data=self.val_generator,
            validation_steps=int(Steps_per_val),
            callbacks=[rlronp, estop]

        )

        model.save('EfficientNetB3.model')

    def VGG19(self):
        model_vgg19 = VGG19(
            weights="imagenet", include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        for layer in model_vgg19.layers[:-1]:
            layer.trainable = False
        top_model = models.Sequential()
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
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=2)

        Steps_per_train = float(self.train_generator.n) / self.train_generator.batch_size
        Steps_per_val = float(self.val_generator.n) / self.val_generator.batch_size

        rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                      factor=0.2,
                                                      mode="min",
                                                      min_lr=1e-6,
                                                      patience=2,
                                                      verbose=1)

        estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 mode="min",
                                                 patience=3,
                                                 verbose=1,
                                                 restore_best_weights=True)
        class_weights = cal_class_weight('./NewDatasets/train_labels.csv')

        model.fit(
            self.train_generator,
            steps_per_epoch=int(Steps_per_train),
            epochs=2,
            verbose=1,
            validation_data=self.val_generator,
            validation_steps=int(Steps_per_val),
            callbacks=[rlronp, estop],
            class_weight=class_weights
        )

        model.save('VGG19.model')

    def SamCNN(self):
        model = models.Sequential()
        # Conv Layer 1
        model.add(Conv2D(filters=32, kernel_size=(5, 5),
                         padding='same', activation=tf.nn.relu, input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Conv Layer 2
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Conv Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Flattening
        model.add(Flatten())

        # FC
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1025, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        # Output
        model.add(Dense(5, activation='softmax'))

        loss = tf.keras.losses.CategoticalCrossentropy(
            label_smoothing=0.0001,
            name='categotical_crossentropy',

        )

        optimizer = optimizers.Adam(learning_rate=1e-4)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['categorical_accuracy']
        )

        model.build((None))

        Steps_per_train = float(self.train_generator.n) / self.train_generator.batch_size
        Steps_per_val = float(self.val_generator.n) / self.val_generator.batch_size

        rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                      factor=0.2,
                                                      mode="min",
                                                      min_lr=1e-6,
                                                      patience=2,
                                                      verbose=1)

        estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 mode="min",
                                                 patience=3,
                                                 verbose=1,
                                                 restore_best_weights=True)

        model.fit(
            self.train_generator,
            steps_per_epoch=int(Steps_per_train),
            epochs=5,
            verbose=1,
            validation_data=self.val_generator,
            validation_steps=int(Steps_per_val),
            callbacks=[rlronp, estop]

        )

        model.save('SamCNN.model')

