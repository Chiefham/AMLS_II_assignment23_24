import keras
import tensorflow as tf
from keras.layers import MaxPooling2D,Flatten,Dense
from keras.layers import Conv2D,BatchNormalization,Dropout


class Models:
    def __init__(self,train_data,train_label,val_data,val_label,test_data,test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
        self.test_data = test_data
        self.test_label = test_label


    def SamCNN(self,LR=0.0001,BS=64,Epochs=50):

        model = keras.models.Sequential()

        # Conv Layer 1
        model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation=tf.nn.relu))
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
            optimizer=tf.keras.optimizers.Adam(lr=LR),
            metrics=['accuracy']
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)

        # Fit
        model.fit(
            x=self.train_data,y=self.train_label,batch_size=BS,epochs=Epochs,
            callbacks=callback,validation_data=(self.val_data,self.val_label)
        )

        # Model Save
        model.save('SamCNN.model')






