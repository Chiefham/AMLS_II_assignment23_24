from keras.applications import EfficientNetB3
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, Activation
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.applications.resnet_v2 import ResNet101V2
from cal_class_weight import cal_class_weight
from hist_plot import hist_plot
import pickle


class Models:
    def __init__(self, target_size_dim, train_generator, val_generator, epochs):
        self.target_size_dim = target_size_dim
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.epochs = epochs
        self.class_weights_dict = cal_class_weight(train_generator)

    def Model(self, base_model, optimizer, metrics=['categorical_accuracy']):
        my_model = Sequential()
        my_model.add(base_model)
        my_model.add(GlobalAveragePooling2D())
        my_model.add(Dense(256))
        my_model.add(BatchNormalization())
        my_model.add(Activation('relu'))
        my_model.add(Dropout(0.3))
        my_model.add(Dense(5, activation='softmax'))
        my_model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(label_smoothing=0.05),
            metrics=metrics
        )
        return my_model

    def EfficientNetB3(self):
        model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(self.target_size_dim, self.target_size_dim, 3),
            drop_connect_rate=0.4
        )
        model.trainable = True

        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        my_model = self.Model(model, optimizer)

        # fit
        early = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=2)
        history = my_model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.epochs,
            callbacks=[early],
            # class_weight=self.class_weights_dict
        )
        my_model.save('EfficientNetB3.model')

        with open('EfficientNetB3_history.pkl', 'wb') as file:
            pickle.dump(history.history, file)
        hist_plot(history, 'EfficientNetB3')

    def VGG19(self):
        model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(self.target_size_dim, self.target_size_dim, 3),
        )
        model.trainable = True

        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        my_model = self.Model(model, optimizer)

        # fit
        early = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=2)

        history = my_model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.epochs,
            callbacks=[early],
            # class_weight=self.class_weights_dict
        )

        my_model.save('VGG19.model')
        with open('VGG19_history.pkl', 'wb') as file:
            pickle.dump(history.history, file)

        hist_plot(history, 'VGG19')

    def ResNet101V2(self):
        model = ResNet101V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.target_size_dim, self.target_size_dim, 3),
        )
        model.trainable = True

        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        my_model = self.Model(model, optimizer)

        # fit
        early = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=2)

        history = my_model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.epochs,
            callbacks=[early],
            # class_weight=self.class_weights_dict
        )
        my_model.save('ResNet101V2.model')
        with open('ResNet101V2_history.pkl', 'wb') as file:
            pickle.dump(history.history, file)

        hist_plot(history, 'ResNet101V2')
