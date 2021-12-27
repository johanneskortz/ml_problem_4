import numpy as np
import cv2

def imagesToGreyScale(RGBArray):

    greyArray = []

    for e, element in enumerate(RGBArray):
        # cv2 needs this format to function properly
        elementConverted = np.array(element, dtype=np.float32)
        greyImage = cv2.cvtColor(elementConverted, cv2.COLOR_BGR2GRAY)
        greyArray.append(greyImage)

    return greyArray