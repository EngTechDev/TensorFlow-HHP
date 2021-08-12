###** Import libraries **###

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

###** Get Data **###

#* Denotes steps added for LEGO50 project

#* Load in LEGO50 dataset
def get_model():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    LEGO50 = keras.Model(inputs, outputs)
    LEGO50.compile(optimizer="adam", loss="mean_squared_error")
    return LEGO50

LEGO50 = get_model()

#* Train the model
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
LEGO50.fit(test_input, test_target)

#* Calling 'save('my_model')' creates a SavedModel folder 'my_model'
LEGO50.save('c:\Dev\AutoGate\envir\LEGO50\dataset')

LEGO50 = keras.models.load_model('c:\Dev\AutoGate\envir\LEGO50\dataset')

# It can be used to reconstruct the model identically



# Define class names to display
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#fashion_mnist = tf.keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = LEGO50

###** Explore Data **###

# Print the shape of the data tensors loaded. Should be:
# 60,000 training 28x28 images and their labels, and 10,0000 testing images and their labels
print('Training data:', train_images.shape, train_labels.shape)
print('Test data:', test_images.shape, test_labels.shape)

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

###** Prepare Data **###

# Scale training and testing image values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Print the image again and notice the values now range from 0 to 1.
# And the image looks the same, just on a different scale.
#show_training_image(img_index)

###** Create Model **###

model = tf.keras.models.Sequential()    # Create a new sequential model
model.add(tf.keras.layers.Flatten(input_shape=(400,400)))     # Keras processing layer - no neurons
model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons
model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense-10-softmax'))     # determines probability of each of the 10 classes

###** Model Structure **###

print('Input Shape:', train_images.shape)
print()
print(model.summary())

###** Compile Model **###

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

###** Train Model **###

train_hist = model.fit(train_images, train_labels, epochs=40)

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

####################### MODEL NEEDS TO IS TRAINED; OPTIMIZE AND IMPROVE #######################

###** Monitor Model Performance **###
import datetime

# Load the tensorboard extension
#% reload_ext tensorboard

# Clear any logs from previous runs
#!rm -rf ./logs/

# Start with a fresh model
model = tf.keras.models.Sequential()    # Create a new sequential model
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
model.add(tf.keras.layers.Dense(10, activation='softmax', name='desne-10-softmax'))     # Determine probability of each of the 10 classes

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

model = tf.keras.models.Sequential()    # Create a new sequential model
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
model.add(tf.keras.layers.Dropout(0.2))     # Dropout 20%
model.add(tf.keras.layers.Dense(10, activation='softmax', name='dens-10-softmax')) #    Determine probability of each of the 10 classes

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

model = tf.keras.models.Sequential()    # Creat a new sequential model
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # Keras processing layer - no neurons
model.add(tf.keras.layers.Dense(128, activation='relu', name='dense-128-relu'))     # 128 neurons connected to pixels
model.add(tf.keras.layers.Dense(10, activation='softmax', name='dense-10-softmax'))   # Determine the probability of each of the 10 classes

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

#!kill 1234   # sometime TensorBoard does not show all data.  If it shows reusing previous instance use kill command listed
#%tensorboard --logdir logs/fit

###** SAVE THE MODEL **###


###** !!INSTRUCTIONS ARE FOR COLAB ONLY!! **###


###** DEPLOY TRAINED MODEL **###


###** !!DEPLOYMENT VARIES DEPENDING ON MODEL USAGE!! **###