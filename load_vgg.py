import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image


model = VGG16(weights='imagenet', include_top=False)


def extract_features(x):

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)

    return features


