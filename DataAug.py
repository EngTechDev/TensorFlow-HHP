import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Use data augmentation before creating a dataset

#BT1
#datagen = ImageDataGenerator(
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')

#img = image.load_img('c:/Dev/AutoGate/envir/ToyMod/Bobtail/BT1.jpeg')
#x = tf.keras.preprocessing.image.img_to_array(img)
#x = x.reshape((1,) + x.shape)

#i = 0
#for batch in datagen.flow(x, batch_size=1,
#                        save_to_dir='c:/Dev/AutoGate/envir/ToyMod/Bobtail', save_prefix='Bobby', save_format='jpeg'):
#    i += 1
#    if i > 20:
#        break

########################
#BT2

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT2.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT3

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT3.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT4

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT4.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT5

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT5.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT6

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT6.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT7

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT7.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT8

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT8.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break

########################
#BT9

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = image.load_img('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail/BT9.jpeg')
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg/Bobtail', save_prefix='Bobby', save_format='jpeg'):
    i += 1
    if i > 20:
        break


# Create a dataset
#ToyMod = keras.preprocessing.image_dataset_from_directory(
#    'c:/Dev/AutoGate/envir/ToyMod', batch_size=18, image_size=(200, 200))

# For demonstration, iterate over the batches yielded by the dataset.
#for data, labels in ToyMod:
#    print(data.shape)
#    print(data.dtype)
#    print(labels.shape)
#    print(labels.dtype)





#from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Example image data, with values in the [0, 255] range
#ToyMod = np.random.randint(0, 256, size=(18, 200, 200, 3))