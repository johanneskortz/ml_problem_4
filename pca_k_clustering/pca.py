from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from helperFunctions.saveAndLoadArrays import *


def applyPCAToGreyImages():

    images = loadGreyImages2D()
    # we need 2 principal components.
    pca = PCA(2)

    pca_result = pca.fit_transform(images)

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))
    c_map = plt.cm.get_cmap('jet', 10)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=15,
                cmap=c_map)
    plt.colorbar()
    plt.xlabel('PC-1'), plt.ylabel('PC-2')
    plt.show()