import os
import sys
import cv2
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='logfile.log', 
                    format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

# // Set path of image dataset //
 
DATA_PATH = "D:/PHD/Leaf Disease/CODE/plantvillage dataset"

_COLOR_IMAGE = os.path.join(DATA_PATH,'color')

total_counts = 0

_path_availabilty = os.path.isdir(DATA_PATH)

if not _path_availabilty:
    logger.error('File not found , Check path correcty,...')
    raise FileNotFoundError

for index, (root , direct, files) in enumerate(os.walk(_COLOR_IMAGE)):
    print(f"{index} {root} counts {len(files)}")
    logger.info(f"{index} {root} counts {len(files)}")
    total_counts += len(files)

logger.info(f"total images in dataset : {total_counts}")
