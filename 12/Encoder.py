from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU

import pandas
import numpy

def import_data():
	FILENAME = "Data.csv"

	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
	del imported_data["^GSPC"]

	data = imported_data.as_matrix()

	data = numpy.log(data)
	data = numpy.diff(data,axis=0)
	data = numpy.absolute(data)

	return (data,data,list(imported_data))

def build_model(activation,input_dim,hidden_dim):
	#Build and Fit Model
	model = Sequential()

	dropoutRate = 0.2

	model.add(Dense(hidden_dim[0],input_dim=input_dim,activation=activation))
	model.add(Dropout(dropoutRate))

	for i in range(1,len(hidden_dim)):
		model.add(Dense(hidden_dim[i],activation=activation))
		model.add(Dropout(dropoutRate))

	#Add Output Layer
	model.add(Dense(input_dim,activation=activation))

	opt = Adam(lr=0.001)
	model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

	return model

def fit_model(model,X,Y):

	model.fit(X,X,batch_size=20,epochs=200,validation_split=0,verbose=2)

	#Calculate Error for each Symbol
	y_pred = model.predict(X)

	print y_pred

	diff = Y - y_pred
	MSE = numpy.sum(numpy.square(diff),axis=0)/X.shape[0]

	return MSE

if __name__ == "__main__":
	X,Y,stocks = import_data()

	activation = ["relu","tanh","sigmoid"]

	ACTIVATION = activation[1]

	HIDDEN_DIM = [50,5,50]
	N_RUNS = 10
	nStocks = X.shape[1]

	results = numpy.empty([N_RUNS,nStocks])

	#Fit Autoencoder multiple times and record results into results file
	for i in range(N_RUNS):
		print "Run %d" % (i)

		model = build_model(ACTIVATION,nStocks,HIDDEN_DIM)
		res = fit_model(model,X,Y)

		results[i] = res

	runSSE = numpy.sum(results,axis=1)
	runMSE = numpy.divide(runSSE,nStocks)

	print "Min: %f\tMax: %f\tAvg: %f" % (min(runMSE),max(runMSE),numpy.mean(runMSE))