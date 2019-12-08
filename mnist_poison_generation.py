# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Convolution2D, Input
import numpy as np
from art.classifiers import KerasClassifier
from art.utils import load_mnist, preprocess
from imageio import imwrite
import pprint
import json


def GenerateModel(backdoor_type="pattern", model_name="poisoned_model", train=True):
    # Read MNIST dataset (x_raw contains the original images):
    (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)

    n_train = np.shape(x_raw)[0]
    num_selection = 10000
    random_selection_indices = np.random.choice(n_train, num_selection)
    x_raw = x_raw[random_selection_indices]
    y_raw = y_raw[random_selection_indices]

    # Poison training data
    perc_poison = .33
    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor(x_raw, y_raw, perc_poison, backdoor_type=backdoor_type)
    x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
    i = 0
    # for x in x_train:
    #     if(is_poison_train[i]):
    #         imwrite("./sample_poisoned_training_data/"+str(i)+"_poisoned.jpg", x)
    #     i += 1

    # Add channel axis:
    x_train = np.expand_dims(x_train, axis=3)

    # Poison test data
    (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = generate_backdoor(x_raw_test, y_raw_test, perc_poison, backdoor_type=backdoor_type)
    x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
    # Add channel axis:
    x_test = np.expand_dims(x_test, axis=3)

    # Shuffle training data so poison is not together
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    is_poison_train = is_poison_train[shuffled_indices]

    # Create Keras convolutional neural network - basic architecture from Keras examples
    # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    ##model.add(Dense(10, activation='softmax'))
    #model.add(Dense(10, name='before_softmax'))
    #model.add(Activation('softmax'))

    ###############
    # input image dimensions
    img_rows, img_cols = 28, 28

    input_shape = (img_rows, img_cols, 1)

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)

    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    # if train:
    # batch_size = 256
    # nb_epoch = 10

    x = Convolution2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    # print("in Model1 x = ", x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    #################

    if train:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        classifier = KerasClassifier(model=model, clip_values=(min_, max_))

        classifier.fit(x_train, y_train, nb_epochs=30, batch_size=128)

        results = open("./models/{0}.txt".format(model_name), "w")

        # Evaluate the classifier on the test set
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        results.write("\nTest accuracy: %.2f%%" % (acc * 100))

        # Evaluate the classifier on poisonous data
        preds = np.argmax(classifier.predict(x_test[is_poison_test]), axis=1)
        acc = np.sum(preds == np.argmax(y_test[is_poison_test], axis=1)) / y_test[is_poison_test].shape[0]
        results.write("\nPoisonous test set accuracy (i.e. effectiveness of poison): %.2f%%" % (acc * 100))

        # Evaluate the classifier on clean data
        preds = np.argmax(classifier.predict(x_test[is_poison_test == 0]), axis=1)
        acc = np.sum(preds == np.argmax(y_test[is_poison_test == 0], axis=1)) / y_test[is_poison_test == 0].shape[0]
        results.write("\nClean test set accuracy: %.2f%%" % (acc * 100))

        results.close()

        # serialize model to JSON
        model_json = model.to_json()
        with open("./models/{0}.json".format(model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("./models/{0}.h5".format(model_name))
        print("Saved model to disk")
    else:
        model.load_weights("./models/{0}.h5".format(model_name))
    return model


def generate_backdoor(x_clean, y_clean, percent_poison, backdoor_type='pattern', sources=np.arange(10),
                      targets=(np.arange(10)+1)%10):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.
    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
    contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """

    max_val = np.max(x_clean)

    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == 'pattern':
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


def add_single_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def add_pattern_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
        x[width - distance - 1, height - distance - 1] = pixel_value
        x[width - distance, height - distance - 2] = pixel_value
        x[width - distance - 2, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


if __name__ == '__main__':
    GenerateModel()
