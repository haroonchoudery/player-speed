from keras.preprocessing.image import ImageDataGenerator
from player_model import get_model
import os
import glob

MODEL_NAME = 'custom_model'
BATCH_SIZE = 16
NUM_EPOCHS = 50

TRAIN_COUNT = 0
TEST_COUNT = 0

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

for dir in listdir_nohidden('data/train'):
    TRAIN_COUNT += len([name for name in os.listdir(dir)])
    
for dir in listdir_nohidden('data/test'):
    TEST_COUNT += len([name for name in os.listdir(dir)])


model = get_model()
model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

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
        target_size=(224, 224),  # all images will be resized to 240x240
        batch_size=BATCH_SIZE,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=TRAIN_COUNT // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=TEST_COUNT // BATCH_SIZE)

try:
    model.save_weights(os.path.join('resources', 'logs', MODEL_NAME+'.h5'))
    print("Model saved ({})".format(MODEL_NAME+'.h5'))
except:
    print("ERROR: Model not saved")