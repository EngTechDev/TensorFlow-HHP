import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pathlib


###** Set Parameters First **###
np.set_printoptions(precision=4)


# Define class names to display
class_names = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun']
classes = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun']
labels = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun']

# Define the parameters of the elements in the data set

train_images=(85, 200, 200, 1)
train_labels=(85,)
test_images=(17, 200, 200, 1)
test_labels=(17,)

x_val = train_images[-17:]
y_val = train_labels[-17:]


training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, dtype='float32')
train_generator = training_datagen.flow_from_directory('C:\Dev\AutoGate\envir\TFCode\ToyMod\ToyModImg', target_size=(200, 200), color_mode='grayscale', batch_size=3)
TRAINING_DIR = pathlib.Path('C:\Dev\AutoGate\envir\TFCode\ToyMod\ToyModImg')

val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, dtype='float32')
validation_generator = val_gen.flow_from_directory('c:/Dev/AutoGate/envir/TFCode/ToyMod/TMOs', target_size=(200, 200), color_mode='grayscale', batch_size=3)
VALIDATION_DIR = pathlib.Path('c:/Dev/AutoGate/envir/TFCode/ToyMod/TMOs')

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, validation_split=0.2, dtype='float32')
images, labels = next(img_gen.flow_from_directory('c:/Dev/AutoGate/envir/TFCode/ToyMod/TMOs', target_size=(200, 200), color_mode='grayscale', classes= labels, batch_size=3))
TMO = pathlib.Path('c:/Dev/AutoGate/envir/TFCode/ToyMod/TMOs')

# Root directory contains a directory for each class:
for item in TMO.glob("*"):
    print(item.name)

# Files in each directory class:
list_ds = tf.data.Dataset.list_files(str(TMO/'*/*'))

for f in list_ds.take(6):
    print(f.numpy())

print("Mandarins1") 

# Extract the label from the path, returning (image, label) pairs:
def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label

labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())

print("Mandarins2")

def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1

for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break

print("Mandarins3")

ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes=((), (None,)))

ds_series

ds_series_batch = ds_series.shuffle(20).padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

print("Mandarins4")

ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(TMO, target_size=(200, 200), color_mode='grayscale', classes=classes, batch_size=3),
    output_types=(tf.float32, tf.float32),
    output_shapes=((3, 200, 200, 1), (3, 6))
)

ds.element_spec

for images, label in ds.take(1):
  print('images.shape: ', images.shape) #(3, 200, 200, 1)
  print('labels.shape: ', labels.shape) #(3, 6)


def ds_gen(ds):
    dsiter = iter(ds)
    try:
        return next(dsiter)
    except StopIteration:
        raise ValueError("iterable 'ds' is empty")

datagen = ds_gen(ds)


###** Train Model **###
# Train the ToyMod for 1 epoch from Numpy data
inputs = tf.keras.Input(shape=(200, 200, 1))
x = tf.keras.layers.Flatten()(inputs)
dense1 = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(6, activation='softmax')(dense1)
ToyMod_Model = tf.keras.Model(inputs=inputs, outputs=outputs)

ToyMod_Model.summary()

# Compile the ToyMod
              
ToyMod_Model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = ToyMod_Model.fit(train_generator, y=None, epochs=10,
                            validation_data=validation_generator)