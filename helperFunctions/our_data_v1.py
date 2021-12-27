from pca_k_clustering.kclustering import *

from boltzmann.first_try import *

def start_ours():

    # Has to be called one time to create the data we want to work with
    #convertAndSaveGreyImagesTo2D()

    # This approach yields the pca which already gives us a insightful
    # result
    #applyPCAToGreyImages()

    # Creates the elbow plot for the kmeans approach.
    # Caution, this takes around 5 minutes to calculate
    #createElbowPlot()

    # Shows a graph with the amount of variance each component
    # contains.
    #pcaPlotVarianceLevels()

    # Shows a plot with the pca result but clustered
    #kClusteringWithPCA()

    start_boltzmann()

