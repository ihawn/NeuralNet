import tensorflow as tf
from tensorflow import keras
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)


#####################
#create word mappings
word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#####################



#####################
#format data with padding
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
#####################



#####################
#Decode word mappings for human reading
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #try to get value at index i. If not there, just say '?'
#####################


#####################
#model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16)) #Group words in a similar way so the computer can determine which ones are similar. (Converge word vectors of similar words closer to each other through context)
model.add(keras.layers.GlobalAveragePooling1D()) #Relax our 16 dimension word vectors into lower dimension to reduce complexity
model.add(keras.layers.Dense(16, activation="relu")) #Classifies the words. Basically all the heavy lifting done here. 16 neurons
model.add(keras.layers.Dense(1, activation="sigmoid")) #Output neuron

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000] #use 10k of our 25k reviews to validate the model accuracy during runtime
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=8, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
#####################



#####################
#Validate results
test_review = test_data[100]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
#####################



#####################
#Save model
#model.save("MovieReviewModel.h5")
#####################



#####################
#Load model
#model = keras.models.load_model("MovieReviewModel.h5")
#####################