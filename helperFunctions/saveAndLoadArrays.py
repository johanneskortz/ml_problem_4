import numpy as np
from pathlib import Path

from helperFunctions.imagesToGreyScale import *

DATA_PATH = Path(
    "D:[ETIT] - Studium/[11.Semester] - ETIT/Machine Learning - Unsupervised Methods/Problems/4) Keeping it clean/data"
)

def loadDataAndConvertToGrey():

    # Get data
    data = loadOriginalData()
    # Convert data to grey
    greyList = imagesToGreyScale(data)
    # Convert to array
    greyArray = np.array(greyList)
    # save the result to save time in the future
    saveGreyImages(greyArray)

def loadOriginalData():
    data = np.load(DATA_PATH / "data.npy")
    return data

def saveGreyImages(greyArray):
    with open("data/greyImagesArray.npy", "wb") as f:
        np.save(f, greyArray)

def loadGreyImages():
    with open("data/greyImagesArray.npy", "rb") as f:
        images = np.load(f)
    return images

def convertAndSaveGreyImagesTo2D():

    greyImages = loadGreyImages()

    greyImages2D = []

    for image in greyImages:
        image1D = np.reshape(image, (70*210,))
        greyImages2D.append(image1D)

    saveGreyImages2D(greyImages2D)


def saveGreyImages2D(greyImages2D):
    with open("data/greyImagesArray2D.npy", "wb") as f:
        np.save(f, greyImages2D)

def loadGreyImages2D():
    with open("data/greyImagesArray2D.npy", "rb") as f:
        images = np.load(f)
    return images