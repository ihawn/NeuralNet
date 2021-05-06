#databases: https://susanqq.github.io/UTKFace/

from time import time
from ParseData import *
from Model import*

#paramaters
seed = int(time())
batch_size = 32
img_height = 200
img_width = 200
split = 0.8
epochs = 100


#parse
img_data = Parse()
age = img_data[0]
sex = img_data[1]
race = img_data[2]
face_matrices = img_data[3]


final_data_sex = Prep_Data(sex, face_matrices, split, seed)
x_train_sex = final_data_sex[0]
x_test_sex = final_data_sex[1]
y_train_sex = final_data_sex[2]
y_test_sex = final_data_sex[3]
class_num_sex = final_data_sex[4]

final_data_race = Prep_Data(race, face_matrices, split, seed)
x_train_race = final_data_race[0]
x_test_race = final_data_race[1]
y_train_race = final_data_race[2]
y_test_race = final_data_race[3]
class_num_race = final_data_race[4]

final_data_age = Prep_Data(age, face_matrices, split, seed)
x_train_age = final_data_age[0]
x_test_age = final_data_age[1]
y_train_age = final_data_age[2]
y_test_age = final_data_age[3]
class_num_age = final_data_age[4]


#train and test
print("Training for sex...")
Init_Train_Test("MaleOrFemale.h5", 'softmax' x_train_sex, y_train_sex, x_test_sex, y_test_sex, seed, class_num_sex, epochs)
print("Training for race...")
Init_Train_Test("Race.h5", 'softmax', x_train_race, y_train_race, x_test_race, y_test_race, seed, class_num_race, epochs)
print("Training for age...")
Init_Train_Test("Age.h5", 'softmax', x_train_age, y_train_age, x_test_age, y_test_age, seed, class_num_age, 30)
print("Training complete")






