import os
import sys
import cv2
import logging
import matplotlib.pyplot as plt


# // Set path of image dataset //
 
DATA_PATH = "D:/PHD/Leaf Disease/CODE/plantvillage dataset"

_COLOR_IMAGE = os.path.join(DATA_PATH,'color')

total_counts = 0

for index, (root , direct, files) in enumerate(os.walk(_COLOR_IMAGE)):
    print(f"{index} {root} counts {len(files)}")
    total_counts += len(files)

