import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

ds_train, ds_info = tfds.load('celeb_a', split='test', shuffle_files=False, with_info=True)
fig = tfds.show_examples(ds_info, ds_train)

sample_size = 2000
ds_train = ds_train.batch(sample_size)
features = next(iter(ds_train.take(1)))
n, h, w, c = features['image'].shape


figure = plt.figure(figsize=(8,6))
sample_images = features['image']
new_image = np.mean(sample_images, axis=0)
plt.imshow(new_image.astype(np.uint8))
plt.axis('off')
plt.show()
