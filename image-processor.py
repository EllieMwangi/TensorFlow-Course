#Load libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load data from Keras dataset library
fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images, test_labels) = fashion_mnist.load_data()

#Display a training image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Flatten the train and test images [0 - 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

#Define model
model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Train model
model.fit(train_images, train_labels, epochs=5)

#Evaluate model performance on test images
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

