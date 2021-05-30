from tensorflow import keras
import numpy as np
#from google.colab import files
from keras.preprocessing import image


model = keras.models.load_model("HorseOrHuman.h5")

fn = 'sexywife.jpg'

path = 'C:/Users/Isaac/Documents/Datasets/Generated/horse-or-human/validation_real/humans/' + fn
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if (classes[0] > 0.5):
    print(fn + " is a human")
else:
    print(fn + " is a horse")
