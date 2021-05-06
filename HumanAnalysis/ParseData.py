from PIL import Image
import pathlib
import os
import numpy as np
from numpy import asarray
from keras.utils import np_utils
import random


def Parse():
    # inialize face images into list
    path = pathlib.Path(r'C:\Users\Isaac\Documents\Datasets')
    faces = list(path.glob('crop_part1\*'))

    # parse data from the filenames
    age = []
    sex = []
    race = []

    for i in range(0, len(faces)):
        filename = os.path.basename(faces[i])
        image_data = str.split(filename, '_')
        age.append([int(image_data[0])])
        sex.append([int(image_data[1])])
        race.append([int(image_data[2])])

    # convert images to matrixes
    face_matrices = []

    print("Converting images to matrices...")
    for i in range(0, len(faces)):  # len(faces)
        face = Image.open(faces[i])
        face_matrices.append(asarray(face))

    return age, sex, race, face_matrices



def Prep_Data(input, face_matrices, split):
    # randomize the input and output arrays
    temp = list(zip(face_matrices, input))
    random.shuffle(temp)
    face_matrices, input = zip(*temp)

    # define x and y data and split it into train and test
    pivot = int(split * len(face_matrices))
    (x_train, y_train), (x_test, y_test) = (face_matrices[0:pivot], input[0:pivot]), (
    face_matrices[pivot + 1:len(face_matrices)], input[pivot + 1:len(face_matrices)])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # scale matrix data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # one hot encode
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    class_num = y_test.shape[1]

    return x_train, x_test, y_train, y_test, class_num
