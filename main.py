import os
import sys
import matplotlib.pyplot as plt


# // Set path of image dataset //
 
DATA_PATH = "D:\PHD\Leaf Disease\CODE\plantvillage dataset"

for root , direct, files in os.walk(DATA_PATH):
    print(root)
    print(direct)