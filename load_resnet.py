import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

if os.path.isfile('models/resnet.h5'):
    
    model = load_model('models/resnet.h5')
else:

    resnet = ResNet50(weights='imagenet', include_top=False)


def extract_feature(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model.predict(img, verbose=0)
    return feature
