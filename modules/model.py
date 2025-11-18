from __future__ import annotations

import os

import tensorflow as tf
import torch
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam


class UNet(object):
    def dice_coef(self, y_true, y_pred):
        if os.environ.get("KERAS_BACKEND") == "torch":
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)
            intersection = torch.sum(y_true * y_pred)
            return 2.0 * intersection / (torch.sum(y_true) + torch.sum(y_pred) + 1)
        else:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true * y_pred)
            return 2.0 * intersection / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

    def dice_coef_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def conv_bn_relu_2(self, sequence, filter_count):
        sequence = Conv2D(filters=filter_count, kernel_size=3, padding="same", kernel_initializer="he_normal")(sequence)
        sequence = Activation("relu")(sequence)
        sequence = Conv2D(filters=filter_count, kernel_size=3, padding="same", kernel_initializer="he_normal")(sequence)
        sequence = Activation("relu")(sequence)
        sequence = BatchNormalization()(sequence)
        return sequence

    def __init__(self, pretrained_weights=None, input_shape=(256, 256, 3)):
        inputs = Input(input_shape)
        conv1 = self.conv_bn_relu_2(inputs, 64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_bn_relu_2(pool1, 128)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_bn_relu_2(pool2, 256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_bn_relu_2(pool3, 512)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.conv_bn_relu_2(pool4, 1024)

        up6 = UpSampling2D(size=(2, 2))(conv5)
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = self.conv_bn_relu_2(merge6, 512)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = self.conv_bn_relu_2(merge7, 256)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = self.conv_bn_relu_2(merge8, 128)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = self.conv_bn_relu_2(merge9, 64)

        conv10 = Conv2D(filters=1, kernel_size=1)(conv9)
        conv10 = Activation("sigmoid")(conv10)

        self.__model = Model(inputs=[inputs], outputs=[conv10])
        self.__model.compile(optimizer=Adam(learning_rate=1e-4), loss=self.dice_coef_loss, metrics=[self.dice_coef])

        if pretrained_weights:
            self.__model.load_weights(pretrained_weights)

    def model(self):
        return self.__model

    def summary(self):
        self.__model.summary()


if __name__ == "__main__":
    model = UNet()
    model.summary()
