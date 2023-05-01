import os
import sys
import cv2
import logging
import numpy as np 
import matplotlib.pyplot as plt

# // set path of image dataset //

DATA_PATH = "D:/PHD/Leaf Disease/CODE/plantvillage dataset"

_COLOR_IMAGE = os.path.join(DATA_PATH,'color')

logging.basicConfig(filename='logfile.log', 
                    format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

logger.info(f'Dataset Path {DATA_PATH}')

# // check File availabilty //

total_counts = 0

_path_availabilty = os.path.isdir(DATA_PATH)

if not _path_availabilty:
    logger.error('File not found , Check path correcty,...')
    raise FileNotFoundError

# // analyze dataset 

unique_specimen = []

for index, (root , direct, files) in enumerate(os.walk(_COLOR_IMAGE)):
    print(f"{index} {root} counts {len(files)}")
    logger.info(f"{index} {root} counts {len(files)}")
    total_counts += len(files)
    if not total_counts == 0:
        unique_specimen.append(root.split('\\')[-1].split('___')[0])

unique_fruits = np.unique(np.array(unique_specimen))

logger.info('Unique Fruit Names')
for _name in unique_fruits:
    print(_name)
    logger.info(_name)

   
logger.info(f"total images in dataset : {total_counts}")

from data_importer import get_images

img_gen = get_images(_COLOR_IMAGE)


n = 0

label_name = list(img_gen.class_indices.keys())[list(img_gen.class_indices.values()).index(n)]



im,info = img_gen.next()

plt.figure()

for i in range(1,26):
    plt.subplot(5, 5, i)
    plt.imshow(im[i]/255)
    plt.title( list(img_gen.class_indices.keys())[np.argmax(info[i],axis=0)])

plt.show(block=False)

