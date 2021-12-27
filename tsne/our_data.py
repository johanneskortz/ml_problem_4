# Importing Modules
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helperFunctions.saveAndLoadArrays import *

def tsne_down_data():
    # Loading dataset
    iris_df = datasets.load_iris()

    images = loadGreyImages2D()

    # Defining Model
    model = TSNE(learning_rate=100)

    # Fitting Model
    transformed = model.fit_transform(images)

    # Plotting 2d t-Sne
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    plt.scatter(x_axis, y_axis)
    plt.show()