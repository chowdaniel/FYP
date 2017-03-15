import pandas
import numpy

from sklearn.decomposition import PCA

#Import data
FILENAME = "Merged.csv"
imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

data = imported_data.as_matrix()

#Build and fit PCA
print "Fitting Model..."
pca = PCA()
pca.fit(data)

components = pca.components_
variance = pca.explained_variance_
ratio = pca.explained_variance_ratio_

output = open("PCA_Ratio.csv","w")
output.write("Component,Variance Ratio\n")

counter = 1
for line in ratio:
	output.write(str(counter))
	output.write(",")
	output.write(str(line))
	output.write("\n")
	counter += 1

output.close()