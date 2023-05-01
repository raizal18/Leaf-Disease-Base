import os
import numpy
import sys
from keras.preprocessing.image import ImageDataGenerator

def get_images(filepath:set) -> object:

    data_gen = ImageDataGenerator(rescale=1.0/255.0)
    image_generator = data_gen.flow_from_directory(filepath)
    return image_generator

def get_segment_mask(filepath):

    pass


def get_train_data(filepath):
    pass


def get_test_data(filepath):
    pass
    
