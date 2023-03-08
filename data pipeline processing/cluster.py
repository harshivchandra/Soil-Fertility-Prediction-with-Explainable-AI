# cluster class - used to cluster data into groups together.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier # since we are only going to use a random forest classifier 
from sklearn.decomposition import PCA
##from loaddata import * (depreciated - not needed anymore)

"""
        In this program, I have decided to implement the clustering algorithm needed by the dataset.
        There are going to be 2 layers to this clustering.
        First Layer clusters each individual parameter of the dataset into 5 classes - very low, low , medium, high, very high
        Second Layer clusters each individual datapoint of the dataset based on previous labels into 3 Soil Fertility metrics - low, medium and high
"""
class cluster:
    """ Used to define the class needed for K-Means clustering data
        This will be used to generate labels for Classification by the random forest model later on.
    """
    def __init__(cluster, no_units=5):
        cluster.k_means_model = KMeans(n_clusters=no_units,random_state=5)
    def split(cluster,dataset):
        cluster.x = dataset
        cluster.xtr, cluster.xt = train_test_split(dataset,test_size = 0.2)
        return cluster
    def train(cluster,dataset):
        cluster.k_means_model.fit(dataset)
    def train_and_predict_labels(cluster):
        cluster.k_means_model.fit(cluster.xtr)
        return cluster.k_means_model.predict(cluster.x)# maybe, maybe not. will have to check this out later.
    def auto(cluster,dataset):
        cluster.split(dataset)
        cluster.train_and_predict_labels()
    def predict(cluster,dataset):
        return cluster.k_means_model.predict(dataset)
    def model_return(cluster):
        return cluster.k_means_model

if(__name__ == '__main__'):

    dd = pd.read_csv("dataset3.csv")
    m = 1
    dp = dd
    pca = PCA()
    o = dd[dd.keys()[1]]
    dd = dd[dd.keys()[3]]
    ##for i in dd:
    ##    o.append(m) not needed anymore
    ##    m = m + 1
    o = np.asarray(o)
    o = np.c_[o,dd]
    imp = IterativeImputer(max_iter=10)
    o = imp.fit_transform(o)
    print(o.shape)
    rr = np.hsplit(o,2)
    ##rr[1] = rr[1].squeeze()
    ##rr[1] = rr[1]/rr[1].max() not needed anymore
    uu = []
    label_list = []
    op = dp[dp.keys()[1]]
    op = np.asarray(op)
    for i in range(2,10):
        uu.append(0)
        dk = dp[dp.keys()[i]]
        m = np.c_[op,dk]
        imp = IterativeImputer(max_iter=10)
        elm = imp.fit_transform(m)
        re = np.hsplit(elm,2)
        uu[i-2] = cluster(5)
        uu[i-2].split(re[1])
        pred = uu[i-2].train_and_predict_labels()
        label_list.append(pred)
        plt.scatter(x=re[0].squeeze(),y=re[1].squeeze(),c=pred, cmap='viridis')
        plt.colorbar()
        plt.show()
    fdss = np.asarray(op)
    for i in label_list:
        fdss = np.c_[fdss,i]
    msw = "pointid ph_cacl2 ph_h20 ec oc caco3 p n k".split(" ")
    dssa = pd.DataFrame(fdss,columns=msw)
    dssa.to_csv("datadump.csv",index=False) # builds labels for data

