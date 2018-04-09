from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def train(train_path,test_path):
    batch_size = 16

    generator = datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    bottleneck_features_train = model.predict_generator(generator, 2000)
    # save the output as a Numpy array
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, 800)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

    train_data = np.load(open('bottleneck_features_train.npy'))
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array([0] * 1000 + [1] * 1000)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * 400 + [1] * 400)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=50,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights('bottleneck_fc_model.h5')