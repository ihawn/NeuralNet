import matplotlib.pyplot as plt
from tensorflow import keras
from ParseData import *
from time import time

#paramaters
seed = int(time())
batch_size = 32
img_height = 200
img_width = 200
split = 0.8

#define catagories
sex_c = ['Male', 'Female']
race_c = ['White', 'Black', 'Asian', 'Indian', 'Other']
age_c = np.array(range(1, 111))

#Parse
img_data = Parse()
age = img_data[0]
sex = img_data[1]
race = img_data[2]
face_matrices = img_data[3]


final_data_sex = Prep_Data(sex, face_matrices, split, seed)
final_data_race = Prep_Data(race, face_matrices, split, seed)
final_data_age = Prep_Data(age, face_matrices, split, seed)

x_test = final_data_sex[1]
y_test_sex = final_data_sex[3]
y_test_race = final_data_race[3]
y_test_age = final_data_age[3]

model_sex = keras.models.load_model("MaleOrFemale.h5")
model_race = keras.models.load_model("Race.h5")
model_age = keras.models.load_model("Age.h5")

#predict
prediction_sex = model_sex.predict(x_test)
prediction_race = model_race.predict(x_test)
prediction_age = model_age.predict(x_test)

for i in range(0, 10):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title("Prediction: " + str(age_c[np.argmax(prediction_age[i])]) + " year old "
              + race_c[np.argmax(prediction_race[i])] + " "
              + sex_c[np.argmax(prediction_sex[i])])

    plt.xlabel("Actual: " + str(np.argmax(y_test_age[i])) + " year old "
               + race_c[np.argmax(y_test_race[i])] + " "
               + sex_c[np.argmax(y_test_sex[i])])
    plt.show()
