import pandas
import numpy

import sklearn.decomposition

def PCA(N):
    #N is the number of components to keep
    FILENAME = "Data.csv"

    imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
    del imported_data["^GSPC"]

    data = imported_data.as_matrix()

    #log returns
    data = numpy.log(data)
    data = numpy.diff(data,axis=0)

    X = data

    #Build and fit PCA
    print "Fitting Model..."
    pca = sklearn.decomposition.PCA()
    pca.fit(X)

    components = pca.components_
    variance = pca.explained_variance_
    ratio = pca.explained_variance_ratio_

    #Reconstruction
    mean = numpy.mean(X,axis=0)
    X_pred = numpy.dot(pca.transform(data)[:,:N],components[:N,:]) + mean

    diff = data - X_pred
    MSE = numpy.sum(numpy.square(diff),axis=0)/X.shape[0]
    avgMSE = numpy.mean(MSE)

    print "Number of Components Used: %d\tMSE: %f" % (N,avgMSE)

if __name__ == "__main__":
	PCA(10)