from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
    return features_train, target_train


def create_model(input_shape):
    optimizer=Adam(lr=0.001)
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                 input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
 
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
 
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
 
    model.add(Dense(84, activation='relu'))
 
    model.add(Dense(10, activation='softmax'))
 
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=20,
               steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model 