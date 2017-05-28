import pandas
import numpy

import sklearn.decomposition

def PCA(N):
	#Import data
	FILENAME = "Data.csv"

	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
	del imported_data["^GSPC"]

	data = imported_data.as_matrix()

	data = numpy.log(data)
	data = numpy.diff(data,axis=0)

	#Build and fit PCA
	print "Fitting Model..."
	pca = sklearn.decomposition.PCA()
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

	#Reconstruction
	mean = numpy.mean(data,axis=0)

	for n in N:
		X_pred = numpy.dot(pca.transform(data)[:,:n],components[:n,:]) + mean

		diff = data - X_pred
		SSE = numpy.sum(numpy.square(diff),axis=0)

		out = open("PCA_SSE" + str(n) + ".csv","w")

		for sse in SSE:
			out.write(str(sse))
			out.write(",")

		out.close()



if __name__ == "__main__":
	PCA([1,5,10])