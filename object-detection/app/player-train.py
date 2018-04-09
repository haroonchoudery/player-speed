from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from init import *
from customNN import customNN
from datetime import datetime
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

def train(training_img_path, test_img_path):

    batch_size = 16
    input_shape = (224,224,3)
    NUM_CLASSES = 2

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            training_img_path,  # this is the target directory
            target_size=(224, 224),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            test_img_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary')
    
    model = customNN(size=input_shape, classes=NUM_CLASSES - 1).model

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=320 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=206 // batch_size)
    model.save_weights(os.path.join('resources', 'logs', 'mobilenet_custom.h5'))
    
if __name__ == '__main__':
    train_path = os.path.join('data', 'train')
    test_path = os.path.join('data', 'test')
    train(train_path, test_path)