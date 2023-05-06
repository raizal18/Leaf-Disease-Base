import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tempfile
from main import img_gen
from sklearn.model_selection import train_test_split
from load_alexnet import extract_alexnet_feature as alexnet
from load_vgg import extract_features as vggnet
from load_resnet import extract_feature as resnet


images = img_gen.filepaths

labels = img_gen.labels

unique_id = img_gen.class_indices


x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.30)


img = Image.open(images[0])

feat1 = alexnet(img)
feat2 = resnet(img)
feat3 = vggnet(img)

print(f"alexnet extracted feature size {feat1.shape}\n resnet extracted feature size {feat2.shape}\n vgg extracted feature size {feat3.shape}\n")

flatten_feature1 = feat1.ravel()
flatten_feature2 = feat2.ravel()
flatten_feature3 = feat3.ravel()


def padding(arr, max_len,val=0):
    new_arr = []
    for i in range(max_len):
        if i<len(arr):
            new_arr.append(arr[i])
        else:
            new_arr.append(0)

    return np.array(new_arr)



import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import cv2

# Define a custom function to read and preprocess the data
def read_data(image_files, label_array):
    # Load and preprocess the images
    images = []
    for image_file in image_files:
        image = Image.open(image_file)
        feat1 = alexnet(image)
        feat2 = resnet(image)
        feat3 = vggnet(image)
        flatten_feature1 = feat1.ravel()
        flatten_feature2 = feat2.ravel()
        flatten_feature3 = feat3.ravel()
        max_len = len(flatten_feature2)
        data = np.expand_dims(np.array([padding(flatten_feature1, max_len),padding(flatten_feature2, max_len),padding(flatten_feature3, max_len)]),axis=0)
    labels = np.array(label_array)
    return data, labels


def data_generator(image_files, label_array, batch_size):

    num_samples = len(image_files)

    index = 0

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    while True:

        start = index
        end = index + batch_size

        if end > num_samples:
            np.random.shuffle(indices)
            start = 0
            end = batch_size

        batch_image_files = [image_files[i] for i in indices[start:end]]
        batch_label_array = label_array[indices[start:end]]
        batch_images, batch_labels = read_data(batch_image_files, batch_label_array)

        index = end

        yield batch_images, batch_labels




# Define the input layers
input1 = tf.keras.layers.Input(shape=(3,flatten_feature2.shape[0]))

# Define the rest of the model architecture
x = tf.keras.layers.LSTM(500)(input1)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(len(unique_id.keys()), activation='softmax')(x)

# Define the model with three inputs and one output
model = tf.keras.models.Model(inputs=input1, outputs=output)




print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = tf.keras.losses.categorical_crossentropy,
               metrics=['accuracy'])



model.fit(data_generator(images, tf.keras.utils.to_categorical(labels), 32),epochs=10,verbose=1)