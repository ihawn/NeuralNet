import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0


#define model architecture/layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation="relu"), #rectifier linear unit
    keras.layers.Dense(10, activation="softmax") #pick values from each neuron so all of them add to 1. Basically a probability of certainty
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5) #train model. An epoch is how many times each neuron will see the same image

test_loss, test_acc = model.evaluate(test_images, test_labels) #test the model against the test images
print("Tested ACC:", test_acc)


#use the model to make predictions
prediction = model.predict(test_images)

#Show some images alongside the model's prediction
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])]) #Pass index from largest value from output probability array into the class names to get what the model thinks the image is
    plt.show()