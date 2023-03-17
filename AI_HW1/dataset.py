import os
import cv2
import numpy as np

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    for file_name in os.listdir(os.path.join(dataPath, "face")):
      img = cv2.imread(os.path.join(dataPath, "face", file_name))[: ,: , 0]
      dataset.append((img, 1))
    for file_name in os.listdir(os.path.join(dataPath, "non-face")):
      img = cv2.imread(os.path.join(dataPath, "non-face", file_name))[:, :, 0]
      dataset.append((img, 0))
    # End your code (Part 1)
    return dataset
