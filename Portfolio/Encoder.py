from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU

import pandas
import numpy
import os

def importData():
	FILENAME = "Data.csv"

	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
	del imported_data["^GSPC"]

	data = imported_data.as_matrix()

	data = numpy.log(data)
	data = numpy.diff(data,axis=0)
	data = numpy.absolute(data)

	return (data,data,list(imported_data))

def buildModel(encoder_activation,decoder_activation,input_dim,hidden_dim):
	#Build and Fit Model
	model = Sequential()
	leakyLayer = LeakyReLU(alpha=0.01)

	if encoder_activation == "leaky":
		model.add(Dense(hidden_dim,input_dim=input_dim))
		model.add(leakyLayer)
	else:
		model.add(Dense(hidden_dim,input_dim=input_dim,activation=encoder_activation))
	model.add(Dropout(0.2))

	if decoder_activation == "leaky":
		model.add(Dense(input_dim))
		model.add(leakyLayer)
	else:
		model.add(Dense(input_dim,activation=decoder_activation))

	opt = Adam(lr=0.001)
	model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

	return model

def fitModel(model,X,Y):

	model.fit(X,X,batch_size=5,epochs=30,validation_split=0,verbose=2)

	#Calculate Error for each Symbol
	y_pred = model.predict(X)

	print y_pred

	diff = Y - y_pred
	MSE = numpy.sum(numpy.square(diff),axis=0)/X.shape[0]

	return MSE

if __name__ == "__main__":
	X,Y,stocks = importData()

	activation = ["relu","tanh","leaky","sigmoid"]

	ENCODER_ACTIVATION = activation[0]
	DECODER_ACTIVATION = activation[0]

	HIDDEN_DIM = 10
	N_RUNS = 20
	nStocks = X.shape[1]

	results = numpy.empty([N_RUNS,nStocks])

	#Fit Autoencoder multiple times and record results into results file
	for i in range(N_RUNS):
		print "Run %d" % (i)

		model = buildModel(ENCODER_ACTIVATION,DECODER_ACTIVATION,nStocks,HIDDEN_DIM)
		res = fitModel(model,X,Y)

		results[i] = res

	sumError = numpy.sum(results,axis=0)
	df = pandas.DataFrame(data=results,columns=stocks)

	df.to_csv("EncoderError.csv")
	portfolio = [x for y,x in sorted(zip(sumError,stocks))]

	output = open("Portfolio.csv","w")
	for stock in portfolio:
		output.write(stock)
		output.write("\n")

	output.close()

	