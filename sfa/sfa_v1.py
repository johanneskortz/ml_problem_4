import sksfa
from helperFunctions.saveAndLoadArrays import *
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def sfa_v1():

    #x = loadGreyImages2D()

    iris = load_iris()

    sfa_transformer = sksfa.SFA(n_components=2).fit(iris.data)
    sfa_2d = sfa_transformer.transform(iris.data)

    plt.scatter(sfa_2d[:, 0], sfa_2d[:, 1], c='g', marker='o')
    plt.title('DBSCAN finds 2 clusters and Noise')
    plt.show()