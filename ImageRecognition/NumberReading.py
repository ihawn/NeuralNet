import tensorflow as tf
import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') >= 0.99):
            print("\nAccuracy is high so cancelling training!")
            self.model.stop_training = True



def train_mnist():
    callbacks = myCallback()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train/255.0
    x_test = x_test/255.0

    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")

    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])


    history = model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])

    return history.epoch, history.history['acc'][-1]

train_mnist()