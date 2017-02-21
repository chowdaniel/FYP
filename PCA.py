import pandas
import numpy

from sklearn.decomposition import PCA

#Import data
FILENAME = "Merged.csv"
imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

data = imported_data.as_matrix()


#Build and fit PCA
pca = PCA(n_components=5)
pca.fit(data)
