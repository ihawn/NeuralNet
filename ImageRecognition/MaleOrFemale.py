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

#paramaters
batch_size = 32
img_height = 200
img_width = 200

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
sex = ['male', 'female']
race = ['White', 'Black', 'Asian', 'Indian', 'Other']

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

#print("Convertinging images to matrices. This will probably take a minute or so...")
#for i in range(0, len(faces)):
#    face = Image.open(faces[i])
#    face_matrices.append(asarray(face))

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(y_test)
print(sex)


#plt.grid(False)
#plt.imshow(face_matrices[120], cmap=plt.cm.binary)
#plt.show()

