from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_datagen_flow = train_datagen.flow_from_directory(
        (path),
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)
    
    return train_datagen_flow


def create_model(input_shape):
    optimizer=Adam(lr=0.001)
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                 input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
 
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
 
    model.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
 
    model.add(Dense(20, activation='relu'))
 
    model.add(Dense(12, activation='softmax'))
 
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=2,
               steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=(test_data),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model 