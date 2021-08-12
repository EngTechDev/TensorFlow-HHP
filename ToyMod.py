import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Define class names to display
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['20ST', '40HC', '40ST', 'Bobtail', 'Chassis', 'Dryrun', 'Stack']

# Create a dataset
ToyMod = keras.preprocessing.image_dataset_from_directory(
    'c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', batch_size=64, image_size=(200, 200))

# For demonstration, iterate over the batches yielded by the dataset.
#for data, labels in ToyMod:
#    print(data.shape)
#    print(data.dtype)
#    print(labels.shape)
#    print(labels.dtype)

###   CONSOLIDATE IMAGES TO SINGLE FORMAT
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(56, 200, 200, 1)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

########################TOYMODINTEG FOLLOWS UNTIL HERE##########################

### BEGIN CREATING THE MODEL
#* added from ToyMod
def get_model():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    ToyMod = keras.Model(inputs, outputs)
    ToyMod.compile(optimizer="adam", loss="mean_squared_error")
    return ToyMod

ToyMod = get_model()

#* Train the model
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
ToyMod.fit(test_input, test_target)

#* Calling 'save('my_model')' creates a SavedModel folder 'my_model'
ToyMod.save('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')

ToyMod = keras.models.load_model('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg')

# It can be used to reconstruct the model identically





#################################COMPILES UNTIL HERE################################

#fashion_mnist = tf.keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = ToyMod.load_data()

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
#* added from ToyMod

dense = keras.layers.Dense(units=16)

inputs = keras.Input(shape=(None, None, None, 2))

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

model = keras.Model(inputs=inputs, outputs=outputs)

data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data.shape)

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.CategoricalCrossentropy())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(ToyMod, epochs=10)

#Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = ToyMod.load_data()

#Build a simple model
inputs = keras.Input(shape=(200, 200))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()
x = layers.Dense(128, activation="relu")
x = layers.Dense(128, activation="relu")
outputs = layers.Dense(10, activation="softmax")
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model for 1 epoch from Numpy data
batch_size = 64
print("Fit on Numpy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)


# Train the model for 1 epoch using a dataset
ToyMod = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(ToyMod, epochs=1)


print(history.history)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(ToyMod, epochs=1)
;###
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(ToyMod, epochs=1, validation_data=val_dataset)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyMod_{epoch}',
        save_freq='epoch')
]
model.fit(ToyMod, epochs=2, callbacks=callbacks)

callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(ToyMod, epochs=2, callbacks=callbacks)

loss, acc = model.evaluate(val_dataset)     # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

predictions = model.predict(val_dataset)
print(predictions.shape)