import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
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


print("Mandarins0")

###** CONSUMING NUMPY ARRAYS GUIDE ON TF **###
###** START of Consuming sets of files **###

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

###** END of Consuming sets of files **###

##### START of Consuming Python Generators #####
def count(stop):
    i=0
    while i<stop:
        yield i
        i += 1

for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.float32, output_shapes=(3, 200, 200, 1))

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

########## END of Consuming Python generators ###################

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


print("Mandarins7")


training = ()
testing = ()
ToyMod = tf.data.Dataset
(training, testing) = ToyMod, ToyMod
(train_images, train_labels) = training, training
(test_images, test_labels) = testing, testing

#train_images = np.asarray(train_images)
#train_labels = np.asarray(train_labels)
#test_images = np.asarray(test_images)
#test_labels = np.asarray(test_labels)
# DirectoryIterator compiles one epoch
ToyMod = tf.keras.preprocessing.image.DirectoryIterator('c:/Dev/AutoGate/envir/TFCode/ToyMod/ToyModImg', image_data_generator=img_gen, target_size=(200, 200), color_mode='grayscale', classes=classes, seed=True, subset='training', dtype='float32')
ToyMod = tf.data.Dataset.from_generator


#class MyModel(tf.keras.Model):
#    def __init__(self):
#        super(MyModel, self).__init__()
#        self.dense1 = tf.keras.layers.Flatten()
#        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
#       self.dense3 = tf.keras.layers.Dense(7, activation='softmax')

#    def call(self, inputs):
#      x = self.dense1(inputs)
#      return self.dense2(x)  

#ToyMod = MyModel()

#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()

###** Train Model **###
# Train the ToyMod for 1 epoch from Numpy data
inputs = tf.keras.Input(shape=(3, 200, 200, 1))
x = tf.keras.layers.Flatten()(inputs)
dense1 = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(7, activation='softmax')(dense1)
ToyMod_Model = tf.keras.Model(inputs=inputs, outputs=outputs)

ToyMod_Model.summary()


ToyModDataset = tf.data.Dataset.list_files('C:\Dev\AutoGate\envir\TFCode\ToyMod\keras_metadata.pb')
for element in ToyModDataset:
    print(element)

print("Mandarins8")

#ToyMod_model = tf.keras.models.Sequential()    # Create a new sequential model
#ToyMod_model.add(tf.keras.layers.InputLayer(input_shape=(200, 200), batch_size=3))
#ToyMod_model.add(tf.keras.layers.Flatten())
#ToyMod_model.add(tf.keras.layers.Dense(32, activation="relu"))
#ToyMod_model.add(tf.keras.layers.Dense(128, activation="relu"))
#ToyMod_model.add(tf.keras.layers.Dense(128, activation="relu"))
#ToyMod_model.add(tf.keras.layers.Dense(7))

#ToyMod_model.summary()





###Evaluate the model                  
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = ToyMod.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)