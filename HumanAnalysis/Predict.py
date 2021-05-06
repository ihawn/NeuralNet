import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import random
from ParseData import *

#paramaters
seed = 10
batch_size = 32
img_height = 200
img_width = 200
split = 0.8

#define catagories
sex_c = ['Male', 'Female']
race_c = ['White', 'Black', 'Asian', 'Indian', 'Other']

#Parse
img_data = Parse()
age = img_data[0]
sex = img_data[1]
race = img_data[2]
face_matrices = img_data[3]

final_data = Prep_Data(sex, face_matrices, split)
x_test = final_data[1]


model_sex = keras.models.load_model("MaleOrFemale.h5")
model_race = keras.models.load_model("Race.h5")

#predict
prediction_sex = model_sex.predict(x_test)
prediction_race = model_race.predict(x_test)

for i in range(20,40):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title("Predicted Sex: " + sex_c[np.argmax(prediction_sex[i])] + "\nPredicted Race: " + race_c[np.argmax(prediction_race[i])])
    plt.show()
