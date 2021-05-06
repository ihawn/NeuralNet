import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from keras.datasets import cifar10
import scipy.misc
from PIL import Image
from numpy import asarray
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import random

#paramaters
seed = 10
batch_size = 32
img_height = 200
img_width = 200
split = 0.8

#inialize face images into list
path = pathlib.Path(r'C:\Users\Isaac\Documents\Datasets')
faces = list(path.glob('crop_part1\*'))

#define validation split as 80 to 20
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#define catagories
sex_c = ['male', 'female']
race_cat = ['White', 'Black', 'Asian', 'Indian', 'Other']

#parse data from the filenames
age = []
sex = []
race = []

for i in range(0, len(faces)):
    filename = os.path.basename(faces[i])
    image_data = str.split(filename, '_')
    age.append([int(image_data[0])])
    sex.append([int(image_data[1])])
    race.append([int(image_data[2])])

#convert images to matrixes
face_matrices = []

print("Convertinging images to matrices. This will probably take a minute or so...")
for i in range(0, len(faces)): #len(faces)
    face = Image.open(faces[i])
    face_matrices.append(asarray(face))

#randomize the input and output arrays
temp = list(zip(face_matrices, sex))
random.shuffle(temp)
face_matrices, sex = zip(*temp)

#define x and y data and split it into train and test
pivot = int(split*len(face_matrices))
(x_train, y_train), (x_test, y_test) = (face_matrices[0:pivot], sex[0:pivot]), (face_matrices[pivot+1:len(face_matrices)], sex[pivot+1:len(face_matrices)])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


#scale matrix data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

#one hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]


#model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dropout(0.2))


model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))

#train
epochs = 25
optimizer = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
np.random.seed(seed)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)

#test
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#save model
model.save("MaleOrFemale.h5")

#predict
prediction = model.predict(x_test)

for i in range(20):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title("Predicted Sex: " + sex_c[np.argmax(prediction[i])])
    plt.show()





