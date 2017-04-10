import pandas
import numpy

#Model Parameters
FILENAME = "Merged.csv"
HIDDEN_LAYER_DIM = 200


#Begin Code
imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
data = imported_data.as_matrix()

data = numpy.log(data)
#data = numpy.diff(data,axis=0)

data_length = data.shape[1]

#print data

print data