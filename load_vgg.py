import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import load_model

if os.path.isfile('models/vgg.h5'):

    model = load_model('models/vgg.h5')

else:
    model = VGG16(weights='imagenet', include_top=False)
    
print(model.summary())

def extract_features(x):

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)

    return features


