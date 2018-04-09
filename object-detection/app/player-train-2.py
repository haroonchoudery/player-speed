
from keras.preprocessing.image import ImageDataGenerator
from player_model import get_model
import os

model = get_model()

model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

batch_size = 16

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
        'data/train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=320 // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=206 // batch_size)

model.save_weights(os.path.join('resources', 'logs', 'custom_model.h5'))