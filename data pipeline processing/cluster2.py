from cluster import *
import pandas as pd
import numpy as np


if(__name__ == "__main__"):
    m = pd.read_csv('datalabelsforparams.csv')
    r = cluster(no_units=3)
    r.split(np.asarray(m[m.keys()[1]]).reshape(m[m.keys()[1]].shape[0],1))
    predicted = r.train_and_predict_labels()
    print(predicted)
    mu = m[m.keys()[0]]

    plt.scatter(m[m.keys()[0]],m[m.keys()[1]],c=predicted)
    plt.colorbar()
    plt.show()
    print(np.unique(predicted))

    giv = "point_id Rel_soil_fertility_category".split(" ")

    ds = pd.DataFrame(data = np.c_[mu,predicted], columns = giv).to_csv('relativesoillabels.csv',index=False)
