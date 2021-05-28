import cv2
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, BatchNormalization, Layer, Reshape, Dropout
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, MaxPooling3D, LSTM, SimpleRNN, RepeatVector
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras import Model
import tensorflow as tf
from configs import *


def process_img(img: np.ndarray):
    saturation = img[..., 1::3]
    value      = img[..., 2::3]
    img = tf.concat([saturation, value], axis=-1)
    img = tf.image.resize(img, (83, 179))
    return img

def layers(data):
    data = Lambda(process_img,
            input_shape=(224, 224, 3), output_shape=(83, 179, 2), name='process')(data)

    data = Conv2D(24, 7, (2, 2), activation=relu, data_format='channels_last', padding='same',
            input_shape=(83, 179, 2), name='conv1a')(data)
    data = Conv2D(24, 3, (1, 1), activation=relu, data_format='channels_last', padding='same',
            input_shape=(42, 90, 24), name='conv1b')(data)
    data = MaxPooling2D((3, 3), (2, 2),           data_format='channels_last', padding='same',
            input_shape=(21, 45, 24), name='pool1c')(data)

    data = Conv2D(32, 3, (2, 2), activation=relu, data_format='channels_last', padding='same',
            input_shape=(11, 23, 24), name='conv2a')(data)
    data = Dropout(0.2)(data)
    data = Conv2D(32, 3, (1, 1), activation=relu, data_format='channels_last', padding='same',
            input_shape=(11, 23, 32),  name='conv2b')(data)
    data = Dropout(0.2)(data)
    data = Conv2D(64, 3, (2, 2), activation=relu, data_format='channels_last', padding='same',
            input_shape=(11, 23, 32),  name='conv2c')(data)
    data = Dropout(0.2)(data)
    data = AveragePooling2D((3, 3), (2, 2),       data_format='channels_last', padding='same',
            input_shape=(6, 12, 64),  name='pool2d')(data)

    data = Flatten(input_shape=(3, 6, 64), name='flatten')(data)
    data = Dense(32,  activation=relu,
            input_shape=(1152,), name='dense1')(data)
    data = Dense(2, activation=tanh,
            input_shape=(32,),  name='dense2')(data)
    return data

def create_model():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output_tensors = layers(input_layer)
    model = tf.keras.Model(input_layer, output_tensors)
    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()