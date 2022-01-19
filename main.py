

import boltzmann.edwin_chen_approach as rbm
import tsne.iris_test as tsne
import tsne.our_data as tsne_own
import pca_k_clustering.pca as pca
import dbscan.dbscan_example as dbscan_example
import dbscan.try_dbscan as dbscan_ours
import dbscan.dbscan_and_pca_on_iris as db_pca_iris

import sfa.sfa_v1 as sfa
import dbscan.dbscan_and_pca_on_our_data as db_pca_our


#rbm.start_edwin_chen()

#tsne.first_tsne_test()
#tsne_own.tsne_down_data()
#pca.applyPCAToGreyImages()

#dbscan_example.start_dbscan()
#dbscan_ours.start_our_dbscan()
#db_pca_iris.launch_pca_and_dbscan()
#db_pca_our.launch_pca_and_dbscan_on_our_data()
sfa.sfa_v1()
