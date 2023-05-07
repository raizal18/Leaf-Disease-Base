import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tempfile
from main import img_gen
from sklearn.model_selection import train_test_split
from load_alexnet import extract_alexnet_feature as alexnet
from load_vgg import extract_features as vggnet
from load_resnet import extract_feature as resnet
from keras.utils import Progbar
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv1D, LSTM,Concatenate,ReLU 
from confusion import confusion
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# set Default Flags  
 
EXTRAXT_FEATURE = False

SAVE_MODEL = False

images = img_gen.filepaths

labels = img_gen.labels

unique_id = img_gen.class_indices

unique_key = {value:key for key, value in unique_id.items()}

def inverse_transform(pred, unique_key : dict = unique_key):
    labels = []
    for instance in pred:
        labels.append(unique_key[np.argmax(instance,axis=0)])
    return np.array(labels)


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



if EXTRAXT_FEATURE == True:

    _, x_train, _, y_train = train_test_split(images, tf.keras.utils.to_categorical(labels),test_size=0.20)

    x_ = []
    y_ = []
    prog = Progbar(len(x_train),width=50)
    for idx,(path, label) in enumerate(zip(x_train,y_train)):
        data,label = np.mean(np.reshape(np.mean(np.squeeze(read_data([path],[label])),axis=0),(32,int(131072/32))),axis=0)
        x_.append(data)
        prog.update(idx)

    np.save('features/labels1.npy', y_train)
    np.save('features/feature1.npy', np.array(x_))
    print('sub 1 completed')

else:
    x_test = train_data = np.load('features/feature.npy')
    y_test = train_labels = np.load('features/labels.npy')



# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((64, 64, 1), input_shape=(4096,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(38, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
# Train the model

model.fit(train_data, train_labels, epochs=10, batch_size=32, callbacks=[tensorboard_callback])

if SAVE_MODEL == True:
    model.save('trained_weights.h5')


# Load model weights for avoid model from scratch

model = tf.keras.models.load_model('trained_weights.h5')

pred = model.predict(x_test)


y_pred = inverse_transform(pred)
y_true = inverse_transform(y_test)


cm = multilabel_confusion_matrix(y_true, y_pred)
# print(classification_report(y_true,y_pred))


met = confusion(y_pred, y_true)

[acc,pre, rec, f1s] = met.metrics()
cm = met.getmatrix()
print('classification report')
print(classification_report(y_pred = met.Y_pred,y_true=met.Y_true))

plt.figure(figsize = (15, 15),dpi = 60)

sns.heatmap(cm, annot=cm,fmt='5d',cbar=False)
cent = 0.5
plt.xticks([0.5 + i for i in range(0,cm.shape[0])],[i for i in range(0,cm.shape[0])])
plt.yticks([0.5 + i for i in range(0,cm.shape[0])],[i for i in range(0,cm.shape[0])])

plt.ylabel('Predicted ')
plt.xlabel('actual ')

plt.show(block=False)

for key , value in unique_key.items():
    print(f'class id {key} : class label -> {value}\n')

print(f"model accuracy : {acc} precision : {pre} recall : {rec} f1_score {f1s}")