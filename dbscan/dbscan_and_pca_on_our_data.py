# Importing Modules
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from helperFunctions.our_data_v1 import *

def launch_pca_and_dbscan_on_our_data():
    # Load Dataset
    iris = load_iris()
    images = loadGreyImages2D()

    # Declaring Model
    dbscan = DBSCAN()

    # Fitting
    dbscan.fit(images)

    # Transoring Using PCA
    pca = PCA(n_components=2).fit(images)
    pca_2d = pca.transform(images)

    # Plot based on Class
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    #plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])


    #c_map = plt.cm.get_cmap('jet', 10)
    #plt.scatter(pca_2d[:, 0], pca_2d[:, 1], s=15,
    #            cmap=c_map)
    #plt.colorbar()
    #plt.xlabel('PC-1'), plt.ylabel('PC-2')
    #plt.show()


    plt.title('DBSCAN finds 2 clusters and Noise')
    plt.show()