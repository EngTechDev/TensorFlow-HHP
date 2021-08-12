import numpy as np
import tensorflow as tf
#from tensorflow import keras, data
#from tensorflow.keras import tf.keras.layers, datasets, models
import os
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from abc import ABCMeta

class MyABC(metaclass=ABCMeta):
    pass

###** Set Parameters First **###
np.set_printoptions(precision=4)


# Define class names to display
class_names = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun', 'Stack']
classes = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun', 'Stack']
labels = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun', 'Stack']

# Define the parameters of the elements in the data set

train_images=(1652, 200, 200, 1)
train_labels=(1652,)
test_images=(413, 200, 200, 1)
test_labels=(413,)

x_val = train_images[-413:]
y_val = train_labels[-413:]
train_images = train_images[:-413]
train_labels = train_labels[:-413]



# Create a dataset
ToyMod = tf.keras.preprocessing.image_dataset_from_directory(
    'c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', class_names = class_names, color_mode = 'grayscale', batch_size=3, image_size=(200, 200), seed=True, validation_split=0.2, subset='training', smart_resize=True)

ToyMod.as_numpy_iterator

print("Mandarins0")
###** CONSUMING NUMPY ARRAYS GUIDE ON TF **###
###** Consuming sets of files **###

ToyMod = tf.data.Dataset
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, validation_split=0.2, dtype='float32')
images, labels = next(img_gen.flow_from_directory('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', target_size=(200, 200), color_mode='grayscale'))
ToyMod = pathlib.Path('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')

# Root directory contains a directory for each class:
for item in ToyMod.glob("*"):
    print(item.name)

# Files in each directory class:
list_ds = tf.data.Dataset.list_files(str(ToyMod/'*/*'))

for f in list_ds.take(9):
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
###** End Module **###
###** Consuming Python Generators **###

##### 7/23 last moved, start here Monday
##### PART of Consuming Python Generators #####
#### Moved UP from save the dataset as a ToyMod_model#######
def count(stop):
    i=0
    while i<stop:
        yield i
        i += 1

for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.float32, output_shapes = (), )
print("Mandarins3")
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
print("Mandarins4")
######## End of continuation section
######## SKIPS DOWN TO FLOWERS; realistic example ImageDataGenerator #######
print(images.dtype, images.shape) #float32 (32, 256, 256, 3)
print(labels.dtype, labels.shape) #float32 (32, 7)

ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(ToyMod, target_size=(200, 200), color_mode='grayscale', classes=classes, batch_size=3),
    output_types=(tf.float32, tf.float32),
    output_shapes=([3, 200, 200, 1], [3, 7])
)

ds.element_spec

for images, label in ds.take(1):
  print('images.shape: ', images.shape) #(3, 256, 256, 1)
  print('labels.shape: ', labels.shape) #(32, 7)

###** Batching dataset elements **###
ToyMod = tf.data.Dataset.range(9)
ToyMod = ToyMod.batch(3)

for batch in ToyMod.take(3):
    print([arr.numpy() for arr in batch])

ToyMod = ToyMod.batch(batch_size=3, drop_remainder=True)

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


print(list(ToyMod.as_numpy_iterator()))
print("Mandarins5")

#ToyMod = tf.keras.preprocessing.image.DirectoryIterator('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', image_data_generator=img_gen, color_mode='grayscale', classes=classes, seed=True, subset='training', dtype='float32')

print("Mandarins6")

#ToyMod = tf.keras.preprocessing.image_dataset_from_directory(
#    'c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', label_mode='int', class_names = class_names, color_mode = 'grayscale', batch_size=3, image_size=(200, 200), seed=True, validation_split=0.2, subset='training', smart_resize=True)


print("Mandarins7")

print("Mandarins8")

training = ()
testing = ()
ToyMod = tf.data.Dataset
(training, testing) = ToyMod, ToyMod
(train_images, train_labels) = training, training
(test_images, test_labels) = testing, testing

#((train_images, train_labels), (test_images, test_labels)) = (training, testing)

#ToyMod = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#for element in ToyMod:
#    print(element)
ToyMod = tf.keras.preprocessing.image.DirectoryIterator('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', image_data_generator=img_gen, target_size=(200, 200), color_mode='grayscale', classes=classes, seed=True, subset='training', dtype='float32')


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

# Center-crop images to 150x150
#x = tf.keras.tf.keras.layers.experimental.preprocessing.CenterCrop(height=150, width=150)(inputs)
#Rescale images to [0, 1]
#x = tf.keras.tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(x)
###** Train Model **###
# Train the ToyMod_model for 1 epoch from Numpy data
#batch_size = 3

ToyMod_model = tf.keras.models.Sequential()    # Create a new sequential model
ToyMod_model.add(tf.keras.layers.InputLayer(input_shape=(200, 200), batch_size=3))
ToyMod_model.add(tf.keras.layers.Flatten())
ToyMod_model.add(tf.keras.layers.Dense(32, activation="relu"))
#ToyMod_model.add(tf.keras.layers.Dense(128, activation="relu"))
#ToyMod_model.add(tf.keras.layers.Dense(128, activation="relu"))
ToyMod_model.add(tf.keras.layers.Dense(7))

ToyMod_model.summary()


# Apply some convolution and pooling tf.keras.layers
#ToyMod_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
#ToyMod_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
#ToyMod_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
#ToyMod_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
#ToyMod_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))

#ToyMod_model.summary()


#x = tf.keras.layers.Dense(128, activation="relu")(x)
#ToyMod_model.add(tf.keras.layers.Dense(128, activation="relu"))
#ToyMod_model.add(tf.keras.layers.Dense(7))

#ToyMod_model.summary()

# Compile the ToyMod_model
ToyMod_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#ToyMod_model.compile(optimizer="adam",
#                loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = ToyMod_model.fit(ToyMod, epochs=10,
                            validation_data=(test_images, test_labels))


###Evaluate the model                  
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = ToyMod_model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# Apply global average pooling to get flat feature vectors
x = tf.keras.layers.GlobalAveragePooling2D()

# Add a dense classifier on top
num_classes = 7
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")

ToyMod_model = tf.keras.Model(num_classes, activation="softmax")

data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")
processed_data = ToyMod_model(data)
print(processed_data.shape)
print(ToyMod_model.summary())



# Start with a fresh ToyMod_model
inputs = tf.keras.Input(shape=(200, 200), batch_size=3)
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)


# Create the ToyMod_model and what the summary looks like
ToyMod_model = tf.keras.Model(inputs, outputs, name="ToyMod_Model")
ToyMod_model.summary()

#keras.utils.plot_model(ToyMod_model, "my_Toy_Model.png")
#keras.utils.plot_model(ToyMod_model, "my_Toy_Model_with_shape_info.png", show_shapes=True)


# Training, evaluation, and inference

# Save the dataset as a ToyMod_model
#ToyMod = tf.data.Dataset

#train_images = tf.reshape(1652, 784).astype("float32") / 255
#test_images = tf.reshape(413, 784).astype("float32") / 255

#train_labels = train_labels.astype("float32")
#test_labels = test_labels.astype("float32")

# Reserve 413 samples for validation


ToyMod = tf.data.Dataset

# Compile the ToyMod_model
ToyMod_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

train_hist = ToyMod_model.fit(x=train_images, y=train_labels, epochs=10,
                            validation_data=(test_images, test_labels))

print(train_hist.history)

print("Fit ToyMod_model on training data")
train_hist = ToyMod_model.fit(train_images, train_labels, batch_size=3, epochs=15, validation_data=(x_val, y_val),)

# Evaluate the ToyMod_model on the test data using `evaluate`
print("Evaluate on test data")
results = ToyMod_model.evaluate(test_images, test_labels, batch_size=3)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = ToyMod_model.predict(test_images[:3])
print("predictions shape:", predictions.shape)

def get_model():
    inputs = keras.Input(shape=(40000,))
    outputs = keras.tf.keras.layers.Dense(7)(inputs)
    ToyMod = keras.Model(inputs, outputs)
    ToyMod.compile(optimizer="adam", loss="mean_squared_error")
    return ToyMod

ToyMod = get_model()


#* Calling 'save('my_model')' creates a SavedModel folder 'my_model'
ToyMod.save('c:/Dev/AutoGate/envir/TFCode/ToyMod')

ToyMod = keras.models.load_model('c:/Dev/AutoGate/envir/TFCode/ToyMod')

####################################################################################################
####################################################################################################
#################################### FOLLOWING FUNCTIONAL API UNTIL HERE #######################################
####################################################################################################
####################################################################################################
from tensorflow.keras.tf.keras.layers.experimental.preprocessing import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

#############################NEXT AFTER FASHION_MNIST PRINTING##########################



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

#################################################################################
((training_data, testing_data)) = ToyMod
(train, test) = ToyMod, ToyMod
(train_images, train_labels), (test_images, test_labels) = (train, train), (test, test)
images, labels = train, train
ToyMod = tf.data.Dataset.from_tensor_slices((images, labels))


###############################################################################

###** Create Model **###
#Build a simple ToyMod_model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = tf.keras.layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.tf.keras.layers.Dense(units=16, activation="relu")(flatten)
x = tf.keras.layers.Dense(64, activation="relu")(dense)
y = tf.keras.layers.Dense(128, activation="relu")(x)
z = tf.keras.layers.Dense(128, activation="relu")(y)
outputs = tf.keras.layers.Dense(10, activation="softmax")(z)
ToyMod_model = keras.Model(inputs, outputs)
print(ToyMod_model.summary())

# Center-crop images to 150x150
#x = tf.keras.tf.keras.layers.experimental.preprocessing.CenterCrop(height=150, width=150)(inputs)
#Rescale images to [0, 1]
#x = tf.keras.tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling tf.keras.layers
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")

# Apply global average pooling to get flat feature vectors
x = tf.keras.layers.GlobalAveragePooling2D()

# Add a dense classifier on top
num_classes = 7
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")

ToyMod_model = keras.Model(num_classes, activation="softmax")

data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")
processed_data = ToyMod_model(data)
print(processed_data.shape)
print(ToyMod_model.summary())



###** Train Model **###
# Train the ToyMod_model for 1 epoch from Numpy data
batch_size = 3
print("Fit on Numpy data")

# Train the ToyMod_model for 1 epoch using a dataset
ToyMod = tf.data.Dataset.from_tensor_slices([train_images, train_labels]).batch(batch_size)
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

# Save the dataset as a ToyMod_model
ToyMod_dataset = ToyMod
##### PART of Consuming Python Generators #####
#### Moved UP from save the dataset as a ToyMod_model#######
def count(stop):
    i=0
    while i<stop:
        yield i
        i += 1
for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.float32, output_shapes = (), )
##### End Module #####

#Get the data as Numpy arrays
# Pack the generator into variables for tensor slices

# Apply dataset transformations to preprocess into NumPy arrays
batched_training_data = tf.constant([1652, 200, 200], shape=(1, 3))
training_data_labeled = tf.constant(["train_images"], shape=(1,1))
batched_training_labels = tf.constant([1652,], shape=(1, 1))
training_label_labeled = tf.constant(["train_labels"], shape=(1,1))
batched_testing_data = tf.constant([413, 200, 200], shape=(1, 3))
testing_data_labeled = tf.constant(["test_images"], shape=(1,1))
batched_testing_labels = tf.constant([413,], shape=(1, 1))
testing_label_labeled = tf.constant(["test_labels"], shape=(1,1))

train_images = tf.data.Dataset.from_tensor_slices([batched_training_data, training_data_labeled]) # combined dataset object
train_labels = tf.data.Dataset.from_tensor_slices([batched_training_labels, training_label_labeled])
test_images = tf.data.Dataset.from_tensor_slices([batched_testing_data, testing_data_labeled]) 
test_labels = tf.data.Dataset.from_tensor_slices([batched_testing_labels, testing_label_labeled]) 
training_data = tf.data.Dataset.zip([train_images, train_labels]) # dataset object separately and combined
testing_data = tf.data.Dataset.zip([test_images, test_labels]) # dataset object separately and combined
ToyMod = tf.data.Dataset.zip([train_images, train_labels], [test_images, test_labels])

((training_data, testing_data)) = ToyMod
(train, test) = ToyMod, ToyMod
(train_images, train_labels), (test_images, test_labels) = (train, train), (test, test)

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


tf.keras.layers.Embedding

ToyMod = tf.data.Dataset.from_tensor_slices([images, labels])
([train_images, train_labels], [test_images, test_labels]) = ToyMod, ToyMod

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
from tensorflow.keras.tf.keras.layers.experimental.preprocessing import Normalization

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

# Save the dataset as a ToyMod_model
#ToyMod = tf.data.Dataset
#(train_images, train_labels), (test_images, test_labels) = (ToyMod)
#((train_images, train_labels), (test_images, test_labels)) = ToyMod

########################COMMENT OUT THIS SECTION##########################


#for element in ToyMod.as_numpy_iterator():
#  print(element)

# Apply dataset transformations to preprocess the data for ToyMod_model
#ToyMod = ToyMod.map(lambda x: x*2)
#list(ToyMod.as_numpy_iterator())
##############################RESUME FOLLWING HERE##################################

# Apply parameters to training and test sets
#ToyMod = tf.data.Dataset.range(5000)
#ToyMod = tf.data.Dataset.batch(56, drop_remainder=False, num_parallel_calls=None, deterministic=None)


###** Explore Data **###


#* Train the ToyMod_model
#test_input = np.random.random((112, 28))
#test_target = np.random.random((112, 1))
#ToyMod.fit(test_input, test_target)



### BEGIN CREATING THE MODEL


################################################################################
#* added from ToyMod
#def get_model():
#    inputs = keras.Input(shape=(32,))
#    outputs = keras.tf.keras.layers.Dense(1)(inputs)
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
#Build a simple ToyMod_model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = tf.keras.layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.tf.keras.layers.Dense(units=16, activation="relu")(flatten)
x = tf.keras.layers.Dense(64, activation="relu")(dense)
y = tf.keras.layers.Dense(128, activation="relu")(x)
z = tf.keras.layers.Dense(128, activation="relu")(y)
outputs = tf.keras.layers.Dense(10, activation="softmax")(z)
ToyMod_model = keras.Model(inputs, outputs)
print(ToyMod_model.summary())



# Center-crop images to 150x150
#x = tf.keras.tf.keras.layers.experimental.preprocessing.CenterCrop(height=150, width=150)(inputs)
#Rescale images to [0, 1]
#x = tf.keras.tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling tf.keras.layers
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")

# Apply global average pooling to get flat feature vectors
x = tf.keras.layers.GlobalAveragePooling2D()

# Add a dense classifier on top
num_classes = 7
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")

ToyMod_model = keras.Model(num_classes, activation="softmax")

data = np.random.randint(0, 256, size=(3, 200, 200, 1)).astype("float32")
processed_data = ToyMod_model(data)
print(processed_data.shape)
print(ToyMod_model.summary())


ToyMod_model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.CategoricalCrossentropy())

ToyMod_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Get the data as Numpy arrays
#(train_images, train_labels), (test_images, test_labels) = ToyMod


# Compile the ToyMod_model
ToyMod_model.compile(optimizer="adam", loss="categorical_crossentropy")

###** Train Model **###
# Train the ToyMod_model for 1 epoch from Numpy data
batch_size = 3
print("Fit on Numpy data")

# Train the ToyMod_model for 1 epoch using a dataset
ToyMod = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
print("Fit on Dataset")

history = ToyMod_model.fit(ToyMod, epochs=10)
print(history.history)

ToyMod_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)
train_hist = ToyMod_model.fit(train_images, train_labels, batch_size=batch_size, epochs=40)

#val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
#history = ToyMod_model.fit(ToyMod, epochs=1, validation_data=val_dataset)
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

test_loss, test_acc = ToyMod_model.evaluate(test_images, test_labels, verbose=0)
print('max training accuracy:', max(train_hist.history['accuracy']), ' test accuracy:', test_acc)

###** Monitor Model Performance **###
import datetime

# Load the tensorboard extension
#% reload_ext tensorboard

# Clear any logs from previous runs
#!rm -rf ./logs/

# Start with a fresh ToyMod_model
inputs = keras.Input(shape=(200, 200), batch_size=3)
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = tf.keras.layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.tf.keras.layers.Dense(units=16, activation="relu")(flatten)
x = tf.keras.layers.Dense(64, activation="relu")(dense)
y = tf.keras.layers.Dense(128, activation="relu")(x)
z = tf.keras.layers.Dense(128, activation="relu")(y)
outputs = tf.keras.layers.Dense(10, activation="softmax")(z)
ToyMod_model = keras.Model(inputs, outputs)
#ToyMod_model = tf.keras.models.Sequential()    # Create a new sequential ToyMod_model
#ToyMod_model.add(tf.keras.tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
#ToyMod_model.add(tf.keras.tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
#ToyMod_model.add(tf.keras.tf.keras.layers.Dense(10, activation='softmax', name='desne-10-softmax'))     # Determine probability of each of the 10 classes

ToyMod_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Add to the fit method the validation/test data. This will cause the training ToyMod_model
# to evaluate itself on the validation/test data on each epoch. This provides per
# epoch data points TensorBoard can plot so we can see the trend.
train_hist = ToyMod_model.fit(train_images, train_labels, epochs=40,
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

#ToyMod_model = tf.keras.models.Sequential()    # Create a new sequential ToyMod_model
#ToyMod_model.add(tf.keras.tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
#ToyMod_model.add(tf.keras.tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
#ToyMod_model.add(tf.keras.tf.keras.layers.Dropout(0.2))     # Dropout 20%
#ToyMod_model.add(tf.keras.tf.keras.layers.Dense(10, activation='softmax', name='dens-10-softmax')) #    Determine probability of each of the 10 classes
#inputs = keras.Input(shape=(200, 200), batch_size=56)
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
flatten = tf.keras.layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.tf.keras.layers.Dense(units=16, activation="relu")(flatten)
x = tf.keras.layers.Dense(64, activation="relu")(dense)
dropo = keras.tf.keras.layers.Dropout(0.2)(x)     # Dropout 20%
y = tf.keras.layers.Dense(128, activation="relu")(dropo)
z = tf.keras.layers.Dense(128, activation="relu")(y)
outputs = tf.keras.layers.Dense(10, activation="softmax")(z)
ToyMod_model = keras.Model(inputs, outputs)

ToyMod_model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

ToyMod_model.fit(x=train_images,
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

#ToyMod_model = tf.keras.models.Sequential()    # Creat a new sequential ToyMod_model
#ToyMod_model.add(tf.keras.tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
#ToyMod_model.add(tf.keras.tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
#ToyMod_model.add(tf.keras.tf.keras.layers.Dense(10, activation='softmax', name='dense-10-softmax'))   # Determine the probability of each of the 10 classes
flatten = tf.keras.layers.Flatten(input_shape=(200, 200))(inputs)
dense = keras.tf.keras.layers.Dense(units=16, activation="relu")(flatten)
x = tf.keras.layers.Dense(64, activation="relu")(dense)
dropo = tf.keras.tf.keras.layers.Dropout(0.2)(x)     # Dropout 20%
y = tf.keras.layers.Dense(128, activation="relu")(dropo)
z = tf.keras.layers.Dense(128, activation="relu")(y)
outputs = tf.keras.layers.Dense(10, activation="softmax")(z)
ToyMod_model = keras.Model(inputs, outputs)

ToyMod_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%M%D-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

ToyMod_model.fit(x=train_images,
          y=train_labels,
          epochs=40,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback, early_stopping_callback])


#callbacks = [
#    keras.callbacks.ModelCheckpoint(
#        filepath='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyMod_{epoch}',
#        save_freq='epoch')
#]
#ToyMod_model.fit(ToyMod, epochs=2, callbacks=callbacks)

#callbacks = [
#    keras.callbacks.TensorBoard(log_dir='./logs')
#]
#ToyMod_model.fit(ToyMod, epochs=2, callbacks=callbacks)

#loss, acc = ToyMod_model.evaluate(validation_data)     # returns loss and metrics
#print("loss: %.2f" % loss)
#print("acc: %.2f" % acc)

#predictions = ToyMod_model.predict(validation_data)
#print(predictions.shape)