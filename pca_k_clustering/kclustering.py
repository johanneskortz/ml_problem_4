import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from timer import *

from helperFunctions.saveAndLoadArrays import *

def createElbowPlot():

    # Load data
    x = loadGreyImages2D()
    # Start measuring the time of execution
    tic()
    # preallocate space
    wcss = []
    # Test 5 cluster numbers to see which is the best one
    for i in range(1, 6):
        model = KMeans(n_clusters=i, init="k-means++")
        model.fit(x)
        wcss.append(model.inertia_)
        toc()
    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 6), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def pcaPlotVarianceLevels():

    x = loadGreyImages2D()

    pca = PCA(2)

    data = pca.fit_transform(x)

    plt.figure(figsize=(10, 10))
    var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    lbls = [str(x) for x in range(1, len(var) + 1)]
    plt.bar(x=range(1, len(var) + 1), height=var, tick_label=lbls)
    plt.show()

def kClusteringWithPCA():

    n_cluster = 3

    x = loadGreyImages2D()
    pca = PCA(2)
    data = pca.fit_transform(x)

    #centers = np.array(model2.cluster_centers_)
    model = KMeans(n_clusters = n_cluster, init = "k-means++")
    label = model.fit_predict(data)
    plt.figure(figsize=(10,10))
    uniq = np.unique(label)
    for i in uniq:
       plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
    #plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='k')
    #This is done to find the centroid for each clusters.
    plt.legend()
    plt.show()