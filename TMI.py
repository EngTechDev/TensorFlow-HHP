import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

###** Set Parameters First **###
np.set_printoptions(precision=4)


# Define class names to display
class_names = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun', 'Stack', 'assets', 'variables']
classes = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun', 'Stack', 'assets', 'variables']

# Define the parameters of the elements in the data set
train_images=(1652, 200, 200)
train_labels=(1652,)
test_images=(413, 200, 200)
test_labels=(413,)

# Create a dataset
#ToyMod = keras.preprocessing.image_dataset_from_directory(
#    'c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', class_names = class_names, color_mode = 'grayscale', batch_size=3, image_size=(200, 200), seed=True, validation_split=0.2, subset='training', smart_resize=True)

###** CONSUMING NUMPY ARRAYS GUIDE ON TF **###
###** Consuming sets of files **###

ToyMod = tf.data.Dataset
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, validation_split=0.2, dtype='float32')
images, labels = next(img_gen.flow_from_directory('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg'))
ToyMod = pathlib.Path('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')

# Root directory contains a directory for each class:
for item in ToyMod.glob("*"):
    print(item.name)

# Files in each directory class:
list_ds = tf.data.Dataset.list_files(str(ToyMod/'*/*'))

for f in list_ds.take(9):
    print(f.numpy())

# Extract the label from the path, returning (image, label) pairs:
def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label

labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())

###** End Module **###
###** Consuming Python Generators **###

##### 7/23 last moved, start here Monday
##### PART of Consuming Python Generators #####
#### Moved UP from save the dataset as a model#######
def count(stop):
    i=0
    while i<stop:
        yield i
        i += 1

for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.float32, output_shapes = (), )
##### End of moved UP section #####
##### Continuation of section added 7/23 still needs to be verified

for count_batch in ds_counter.repeat().batch(3).take(3):
  print(count_batch.numpy())

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
######## End of continuation section
######## SKIPS DOWN TO FLOWERS; realistic example ImageDataGenerator #######
print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(ToyMod, color_mode='grayscale', classes=classes, batch_size=3),
    output_types=(tf.float32, tf.float32),
    output_shapes=([3, 256, 256, 1], [3, 9])
)

ds.element_spec

for images, label in ds.take(1):
  print('images.shape: ', images.shape)
  print('labels.shape: ', labels.shape)

###** Batching dataset elements **###
ToyMod = tf.data.Dataset.range(9)
ToyMod = ToyMod.batch(3)

for batch in ToyMod.take(3):
    print([arr.numpy() for arr in batch])

ToyMod = ToyMod.batch(batch_size=3, drop_remainder=True)

print(list(ToyMod.as_numpy_iterator()))
ToyMod = tf.keras.preprocessing.image.DirectoryIterator('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', image_data_generator=img_gen, color_mode='grayscale', classes=classes, seed=True, subset='training', dtype='float32')

#################################### COMPILES UNTIL HERE #######################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#################################### PLUGGED IN FROM HERE #######################################
####################################################################################################
####################################################################################################
#ToyMod = tf.data.Dataset
#(training, testing) = ToyMod_dataset
#((train_images, train_labels), (test_images, test_labels)) = (training, testing)
# Start with a fresh model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.layers.Dense(units=16, activation="relu")(flatten)
x = layers.Dense(64, activation="relu")(dense)
y = layers.Dense(128, activation="relu")(x)
z = layers.Dense(128, activation="relu")(y)
outputs = layers.Dense(10, activation="softmax")(z)

# Create the model and what the summary looks like
model = keras.Model(inputs, outputs, name="ToyMod_Model")
model.summary()

keras.utils.plot_model(model, "my_Toy_Model.png")
keras.utils.plot_model(model, "my_Toy_Model_with_shape_info.png", show_shapes=True)

# Training, evaluation, and inference
#Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.ToyMod.load_data()

x_train = x_train.reshape(1652, 784).astype("float32") / 255
x_test = x_test.reshape(413, 784).astype("float32") / 255

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(ToyMod, epochs=10)
print(history.history)


train_hist = model.fit(train_images, train_labels, batch_size=batch_size, epochs=40)

####################################################################################################
####################################################################################################
#################################### FOLLOWING FUNCTIONAL API UNTIL HERE #######################################
####################################################################################################
####################################################################################################

((training_data, testing_data)) = ToyMod_dataset
(train, test) = ToyMod, ToyMod
(train_images, train_labels), (test_images, test_labels) = (train, train), (test, test)
images, labels = train, train
ToyMod = tf.data.Dataset.from_tensor_slices((images, labels))

# Save the dataset as a model
#ToyMod = tf.data.Dataset
#(train_images, train_labels), (test_images, test_labels) = (ToyMod)
#((train_images, train_labels), (test_images, test_labels)) = ToyMod
#* added from ToyMod
def get_model():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    ToyMod = keras.Model(inputs, outputs)
    ToyMod.compile(optimizer="adam", loss="mean_squared_error")
    return ToyMod

#ToyMod = get_model()


#* Calling 'save('my_model')' creates a SavedModel folder 'my_model'
#ToyMod.save('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')

#ToyMod = keras.models.load_model('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')


#* added from ToyMod
###############################################################################

###** Create Model **###
#Build a simple model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.layers.Dense(units=16, activation="relu")(flatten)
x = layers.Dense(64, activation="relu")(dense)
y = layers.Dense(128, activation="relu")(x)
z = layers.Dense(128, activation="relu")(y)
outputs = layers.Dense(10, activation="softmax")(z)
model = keras.Model(inputs, outputs)
print(model.summary())

# Center-crop images to 150x150
#x = tf.keras.layers.experimental.preprocessing.CenterCrop(height=150, width=150)(inputs)
#Rescale images to [0, 1]
#x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
x = layers.MaxPooling2D(pool_size=(3, 3))
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")
x = layers.MaxPooling2D(pool_size=(3, 3))
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()

# Add a dense classifier on top
num_classes = 7
outputs = layers.Dense(num_classes, activation="softmax")

model = keras.Model(num_classes, activation="softmax")

data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")
processed_data = model(data)
print(processed_data.shape)
print(model.summary())



###** Train Model **###
# Train the model for 1 epoch from Numpy data
batch_size = 3
print("Fit on Numpy data")

# Train the model for 1 epoch using a dataset
ToyMod = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
print("Fit on Dataset")


#################################### ABOVE THIS PLUGGED IN FROM LOWER #######################################
####################################################################################################
####################################################################################################
########################COMMENT OUT THIS SECTION##########################
#ToyMod_training(train_images, train_labels) = (images, labels)
#(test_images, test_labels) = ToyMod
print(list(ToyMod.as_numpy_iterator()))

#for element in ToyMod.as_numpy_iterator():
#  print(element)

#print(ToyMod.reduce(0, lambda state, value: state + value).numpy())

# Save the dataset as a model
ToyMod_dataset = ToyMod
##### PART of Consuming Python Generators #####
#### Moved UP from save the dataset as a model#######
def count(stop):
    i=0
    while i<stop:
        yield i
        i += 1

for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.float32, output_shapes = (), )
##### End Module #####

# Apply dataset transformations to preprocess into NumPy arrays
batched_training_data = tf.constant([1652, 200, 200], shape=(1, 3))
training_data_labeled = tf.constant(["train_images"], shape=(1,1))
batched_training_labels = tf.constant([1652,], shape=(1, 1))
training_label_labeled = tf.constant(["train_labels"], shape=(1,1))
batched_testing_data = tf.constant([413, 200, 200], shape=(1, 3))
testing_data_labeled = tf.constant(["test_images"], shape=(1,1))
batched_testing_labels = tf.constant([413,], shape=(1, 1))
testing_label_labeled = tf.constant(["test_labels"], shape=(1,1))

train_images = tf.data.Dataset.from_tensor_slices((batched_training_data, training_data_labeled)) # combined dataset object
train_labels = tf.data.Dataset.from_tensor_slices((batched_training_labels, training_label_labeled))
test_images = tf.data.Dataset.from_tensor_slices((batched_testing_data, testing_data_labeled)) 
test_labels = tf.data.Dataset.from_tensor_slices((batched_testing_labels, testing_label_labeled)) 
training_data = tf.data.Dataset.zip((train_images, train_labels)) # dataset object separately and combined
testing_data = tf.data.Dataset.zip((test_images, test_labels)) # dataset object separately and combined
#ToyMod_dataset = tf.data.Dataset.zip(features_dataset, labels_dataset)

((training_data, testing_data)) = ToyMod_dataset

###** ALTTERNATE SET TO TRY IF ZIPPING MULTIPLES DOES NOT WORK **###

#ToyMod_training_data = tf.data.Dataset.from_tensor_slices((batched_training_data, training_data_labeled)) # combined dataset object
#ToyMod_training_labels = tf.data.Dataset.from_tensor_slices((batched_training_labels, training_label_labeled))
#ToyMod_testing_data = tf.data.Dataset.from_tensor_slices((batched_testing_data, testing_data_labeled)) 
#ToyMod_testing_labels = tf.data.Dataset.from_tensor_slices((batched_testing_labels, testing_label_labeled)) 
#features_dataset = tf.data.Dataset.zip(ToyMod_training_data, ToyMod_training_labels) # dataset object separately and combined
#labels_dataset = tf.data.Dataset.zip(ToyMod_testing_data, ToyMod_testing_labels) # dataset object separately and combined
#ToyMod_dataset = tf.data.Dataset.zip(features_dataset, labels_dataset)


for element in ToyMod.as_numpy_iterator():
  print(element)


#ToyMod = tf.data.Dataset
#(training, testing) = ToyMod_dataset
#((train_images, train_labels), (test_images, test_labels)) = (training, testing)

# For demonstration, iterate over the batches yielded by the dataset.
for images, labels in ToyMod:
    print(images.shape)
    print(images.dtype)
    print(labels.shape)
    print(labels.dtype)

# Print the shape of the data tensors loaded. Should be:
# 1652 training 200x200 images and their labels, and 413 testing images and their labels
#print('Training data:', train_images.shape, train_labels.shape)
#print('Test data:', test_images.shape, test_labels.shape)

###   CONSOLIDATE IMAGES TO SINGLE FORMAT
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

#############################NEXT AFTER FASHION_MNIST PRINTING##########################

###** Inspect Data **###

def show_training_image(index):
    img_label = str(train_labels[index]) + ' (' + class_names[train_labels[index]] + ')'
    plt.figure()
    plt.title('Image Label ' + img_label)
    plt.imshow(train_images[index], cmap='gray')    # data is grayscale, but displays in color with out cmap='gray'
    plt.colorbar()
    plt.show()

    img_index = 100
    show_training_image(img_index)

######################IF data does not compile TRY labels instead of train_labels################
######################test_images has one value in tuple; may need to normalize test data################
###** Prepare Data **###

# Scale training and testing image values
train_images = train_images / 255.0
test_images = test_images / 255.0
#image_paths = image_paths / 255.0

# Print the image again and notice the values now range from 0 to 1.
# And the image looks the same, just on a different scale.
#show_training_image(img_index)

# Optional added 7/16 to see if  these variables are assigned properly
print(train_images.dtype)
print(test_images.dtype)
print(train_images.shape)
print(test_images.shape)

########################RESUME FOLLOWING HERE###################

# Save the dataset as a model
#ToyMod = tf.data.Dataset
#(train_images, train_labels), (test_images, test_labels) = (ToyMod)
#((train_images, train_labels), (test_images, test_labels)) = ToyMod

########################COMMENT OUT THIS SECTION##########################
#ToyMod_training(train_images, train_labels) = (images, labels)
#(test_images, test_labels) = ToyMod

# Apply dataset transformations to preprocess into NumPy arrays
batched_training_data = tf.constant([1652, 200, 200], shape=(1, 3))
training_data_labeled = tf.constant([train_images], shape=(1,1))
batched_training_labels = tf.constant([1652,], shape=(1, 1))
training_label_labeled = tf.constant([train_labels], shape=(1,1))
batched_testing_data = tf.constant([413, 200, 200], shape=(1, 3))
testing_data_labeled = tf.constant([test_images], shape=(1,1))
batched_testing_labels = tf.constant([413,], shape=(1, 1))
testing_label_labeled = tf.constant([test_labels], shape=(1,1))

ToyMod_training_data = tf.data.Dataset.from_tensor_slices((batched_training_data, training_data_labeled)) # combined dataset object
ToyMod_training_labels = tf.data.Dataset.from_tensor_slices((batched_training_labels, training_label_labeled))
ToyMod_testing_data = tf.data.Dataset.from_tensor_slices((batched_testing_data, testing_data_labeled)) 
ToyMod_testing_labels = tf.data.Dataset.from_tensor_slices((batched_testing_labels, testing_label_labeled)) 
features_dataset = tf.data.Dataset.zip(ToyMod_training_data, ToyMod_training_labels) # dataset object separately and combined
labels_dataset = tf.data.Dataset.zip(ToyMod_testing_data, ToyMod_testing_labels) # dataset object separately and combined
ToyMod_dataset = tf.data.Dataset.zip(features_dataset, labels_dataset)



###** ALTTERNATE SET TO TRY IF ZIPPING MULTIPLES DOES NOT WORK **###
batched_training_data = tf.constant([1652, 200, 200], shape=(1, 3))
training_data_labeled = tf.constant(["train_images"], shape=(1,1))
batched_training_labels = tf.constant([1652,], shape=(1, 1))
training_label_labeled = tf.constant(["train_labels"], shape=(1,1))
batched_testing_data = tf.constant([413, 200, 200], shape=(1, 3))
testing_data_labeled = tf.constant(["test_images"], shape=(1,1))
batched_testing_labels = tf.constant([413,], shape=(1, 1))
testing_label_labeled = tf.constant(["test_labels"], shape=(1,1))

train_images = tf.data.Dataset.from_tensor_slices((batched_training_data, training_data_labeled)) # combined dataset object
train_labels = tf.data.Dataset.from_tensor_slices((batched_training_labels, training_label_labeled))
test_images = tf.data.Dataset.from_tensor_slices((batched_testing_data, testing_data_labeled)) 
test_labels = tf.data.Dataset.from_tensor_slices((batched_testing_labels, testing_label_labeled)) 
features_dataset = tf.data.Dataset.zip(ToyMod_training_data, ToyMod_training_labels) # dataset object separately and combined
labels_dataset = tf.data.Dataset.zip(ToyMod_testing_data, ToyMod_testing_labels) # dataset object separately and combined
ToyMod = tf.data.Dataset.zip(features_dataset, labels_dataset)

#training_batched_data = tf.constant([[[1652, 200, 200], [1652,]],[[413, 200, 200], [413,]]], shape=(2, 2))
#training_batched_labels = tf.constant([['train_images', 'train_labels'], ['test_images', 'test_labels']], shape=(2,2))
#ToyMod = tf.data.Dataset.from_tensor_slices((training_batched_data, training_batched_labels)) # combined dataset object
#features_dataset = tf.data.Dataset.from_tensor_slices(training_batched_data)
#labels_dataset = tf.data.Dataset.from_tensor_slices(training_batched_labels)
#ToyMod_dataset = tf.data.Dataset.zip((features_dataset, labels_dataset)) # dataset object separately and combined

#for element in ToyMod.as_numpy_iterator():
#  print(element)

# Apply dataset transformations to preprocess the data for model
#ToyMod = ToyMod.map(lambda x: x*2)
#list(ToyMod.as_numpy_iterator())
##############################RESUME FOLLWING HERE##################################

# Apply parameters to training and test sets
#ToyMod = tf.data.Dataset.range(5000)
#ToyMod = tf.data.Dataset.batch(56, drop_remainder=False, num_parallel_calls=None, deterministic=None)


###** Explore Data **###


#* Train the model
#test_input = np.random.random((112, 28))
#test_target = np.random.random((112, 1))
#ToyMod.fit(test_input, test_target)



### BEGIN CREATING THE MODEL


################################################################################
#* added from ToyMod
#def get_model():
#    inputs = keras.Input(shape=(32,))
#    outputs = keras.layers.Dense(1)(inputs)
#    ToyMod = keras.Model(inputs, outputs)
#    ToyMod.compile(optimizer="adam", loss="mean_squared_error")
#    return ToyMod

#ToyMod = get_model()


#* Calling 'save('my_model')' creates a SavedModel folder 'my_model'
#ToyMod.save('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')

#ToyMod = keras.models.load_model('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')


#* added from ToyMod
###############################################################################

###** Create Model **###
#Build a simple model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.layers.Dense(units=16, activation="relu")(flatten)
x = layers.Dense(64, activation="relu")(dense)
y = layers.Dense(128, activation="relu")(x)
z = layers.Dense(128, activation="relu")(y)
outputs = layers.Dense(10, activation="softmax")(z)
model = keras.Model(inputs, outputs)
print(model.summary())



# Center-crop images to 150x150
#x = tf.keras.layers.experimental.preprocessing.CenterCrop(height=150, width=150)(inputs)
#Rescale images to [0, 1]
#x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
x = layers.MaxPooling2D(pool_size=(3, 3))
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")
x = layers.MaxPooling2D(pool_size=(3, 3))
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()

# Add a dense classifier on top
num_classes = 7
outputs = layers.Dense(num_classes, activation="softmax")

model = keras.Model(num_classes, activation="softmax")

data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")
processed_data = model(data)
print(processed_data.shape)
print(model.summary())


model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy())

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Get the data as Numpy arrays
#(train_images, train_labels), (test_images, test_labels) = ToyMod


# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

###** Train Model **###
# Train the model for 1 epoch from Numpy data
batch_size = 3
print("Fit on Numpy data")

# Train the model for 1 epoch using a dataset
ToyMod = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
print("Fit on Dataset")

history = model.fit(ToyMod, epochs=10)
print(history.history)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
train_hist = model.fit(train_images, train_labels, batch_size=batch_size, epochs=40)

#val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
#history = model.fit(ToyMod, epochs=1, validation_data=val_dataset)
def plot_acc(hist):
    # Plot the accuracy
    plt.title('Accuracy History')
    plt.plot(hist.history['accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.show()

def plot_loss(hist):
    # Plot the loss
    plt.title('Loss History')
    plt.plot(hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

plot_loss(train_hist)
plot_acc(train_hist)

###** Evaluate Trained Model **###

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('max training accuracy:', max(train_hist.history['accuracy']), ' test accuracy:', test_acc)

###** Monitor Model Performance **###
import datetime

# Load the tensorboard extension
#% reload_ext tensorboard

# Clear any logs from previous runs
#!rm -rf ./logs/

# Start with a fresh model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.layers.Dense(units=16, activation="relu")(flatten)
x = layers.Dense(64, activation="relu")(dense)
y = layers.Dense(128, activation="relu")(x)
z = layers.Dense(128, activation="relu")(y)
outputs = layers.Dense(10, activation="softmax")(z)
model = keras.Model(inputs, outputs)
#model = tf.keras.models.Sequential()    # Create a new sequential model
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
#model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
#model.add(tf.keras.layers.Dense(10, activation='softmax', name='desne-10-softmax'))     # Determine probability of each of the 10 classes

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Add to the fit method the validation/test data. This will cause the training model
# to evaluate itself on the validation/test data on each epoch. This provides per
# epoch data points TensorBoard can plot so we can see the trend.
train_hist = model.fit(train_images, train_labels, epochs=40,
            validation_data=(test_images, test_labels),
            callbacks=[tensorboard_callback])

#!kill 1234     # Sometimes TensorBoard does not show all data. If it shows reusing previous instance use kill command listed
#%tensorboard --logdir logs/fit


###** !!REDUCING MODEL COMPLEXITY NOT INCLUDED!! **###


###** Random Dropout **###

#   Load the tensorboard extension
#% reload_ext tensorboard_callback

#   Clear any logs from previous layer
#!rm -rf ./logs/

#model = tf.keras.models.Sequential()    # Create a new sequential model
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
#model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
#model.add(tf.keras.layers.Dropout(0.2))     # Dropout 20%
#model.add(tf.keras.layers.Dense(10, activation='softmax', name='dens-10-softmax')) #    Determine probability of each of the 10 classes
#inputs = keras.Input(shape=(200, 200), batch_size=56)
#rescale = layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.layers.Dense(units=16, activation="relu")(flatten)
x = layers.Dense(64, activation="relu")(dense)
dropo = keras.layers.Dropout(0.2)(x)     # Dropout 20%
y = layers.Dense(128, activation="relu")(dropo)
z = layers.Dense(128, activation="relu")(y)
outputs = layers.Dense(10, activation="softmax")(z)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=train_images,
        y=train_labels,
        epochs=40,
        validation_data=(test_images, test_labels),
        callbacks=[tensorboard_callback])

#!kill 1234     # Sometime Tensorboard does not show all data. If it shows reusing previous instance use kill command listed
#%tensorboard --logdir logs/fit

###** Implement Early Stopping **###

#   Load the tensorboard extension
#% reload_ext tensorboard

# Clear any logs from previous runs
#!rm -rf ./logs/

#model = tf.keras.models.Sequential()    # Creat a new sequential model
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
#model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
#model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense-10-softmax'))   # Determine the probability of each of the 10 classes
flatten = layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.layers.Dense(units=16, activation="relu")(flatten)
x = layers.Dense(64, activation="relu")(dense)
dropo = tf.keras.layers.Dropout(0.2)(x)     # Dropout 20%
y = layers.Dense(128, activation="relu")(dropo)
z = layers.Dense(128, activation="relu")(y)
outputs = layers.Dense(10, activation="softmax")(z)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%M%D-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

model.fit(x=train_images,
          y=train_labels,
          epochs=40,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback, early_stopping_callback])


#callbacks = [
#    keras.callbacks.ModelCheckpoint(
#        filepath='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyMod_{epoch}',
#        save_freq='epoch')
#]
#model.fit(ToyMod, epochs=2, callbacks=callbacks)

#callbacks = [
#    keras.callbacks.TensorBoard(log_dir='./logs')
#]
#model.fit(ToyMod, epochs=2, callbacks=callbacks)

#loss, acc = model.evaluate(validation_data)     # returns loss and metrics
#print("loss: %.2f" % loss)
#print("acc: %.2f" % acc)

#predictions = model.predict(validation_data)
#print(predictions.shape)