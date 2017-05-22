from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU

import pandas
import numpy
import os

def importData():
	start1 = "2014-01-02"
	start2 = "2014-06-02"
	start3 = "2014-11-03"
	start4 = "2015-04-01"
	start5 = "2015-09-01"

	end1 = "2014-05-30"
	end2 = "2014-10-31"
	end3 = "2015-03-31"
	end4 = "2015-08-31"
	end5 = "2015-12-31"

	start6 = "2016-01-04"
	end6 = "2016-12-30"

	FILENAME = "Data.csv"
	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

	sample = imported_data.loc[start1:end5]
	validation = imported_data.loc[start6:end6]

	return (sample,validation)

def buildModel(input_dim,leaky=False):
	model = Sequential()

	leakyLayer = LeakyReLU(alpha=0.01)

	activation = "relu"
	dropout_rate - 0.5

	if leaky:
		model.add(Dense(output_dim=4,input_dim=input_dim))
		model.add(leakyLayer)
	else:
		model.add(Dense(output_dim=4,input_dim=input_dim,activation=activation))
	model.add(Dropout(dropout_rate))

	if leaky:
		model.add(Dense(output_dim=2))
		model.add(leakyLayer)
	else:
		model.add(Dense(output_dim=2,activation=activation))
	model.add(Dropout(dropout_rate))
	model.add(Dense(output_dim=1))

	return model

def Replicating(sample,validation,parameters):
	s_res = pandas.DataFrame(index=sample.index[1:])
	v_res = pandas.DataFrame(index=validation.index[1:])

	s_res["^GSPC"] = numpy.diff(numpy.log(sample.as_matrix(columns=["^GSPC"])),axis=0)
	v_res["^GSPC"] = numpy.diff(numpy.log(validation.as_matrix(columns=["^GSPC"])),axis=0)

	counter = 0
	for params in parameters:
		counter += 1
		input_dim = params[0] + params[1]
		#Import list of stocks sorted by increasing SSE
		symbols = open("Portfolio.csv","r")

		stocks = []
		for symbol in symbols:
			if symbol != "\n":
				stocks.append(symbol.replace("\n",""))
		symbols.close()
		
		#Select the stocks used for replicating portfolio
		chosen_stocks = []
		#Stocks with most communal info
		for i in range(params[0]):
			chosen_stocks.append(stocks[i])
		#Stocks with least communal info
		for i in range(params[1]):
			chosen_stocks.append(stocks[-i-1])
		chosen_stocks.append("^GSPC")


		#Calibration Phase====================================================

		#Import data for stocks
		data = pandas.DataFrame()

		for stock in chosen_stocks:
			prices = sample[stock]
			data[stock] = prices

		#Extract X and Y as numpy arrays
		X = data.as_matrix(columns=chosen_stocks[0:-1])
		Y = data.as_matrix(columns=chosen_stocks[-1:])

		#log prices
		X = numpy.log(X)
		Y = numpy.log(Y)

		#log returns
		X = numpy.diff(X,axis=0)
		Y = numpy.diff(Y,axis=0)

		#Build and fit Deep Network
		model = buildModel(input_dim,leaky=False)

		opt = Adam(lr=0.001)
		model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

		model.fit(X,Y,batch_size=40,nb_epoch=100,validation_split=0,verbose=0)

		Y_pred = model.predict(X)
		s_res[counter] = Y_pred

		#Validation Phase====================================================

		#Import data for stocks
		data = pandas.DataFrame()

		for stock in chosen_stocks:
			prices = validation[stock]
			data[stock] = prices

		#Extract X and Y as numpy arrays
		X = data.as_matrix(columns=chosen_stocks[0:-1])
		Y = data.as_matrix(columns=chosen_stocks[-1:])

		#log prices
		X = numpy.log(X)
		Y = numpy.log(Y)

		#log returns
		X = numpy.diff(X,axis=0)
		Y = numpy.diff(Y,axis=0)

		Y_pred = model.predict(X)
		v_res[counter] = Y_pred

	s_res.to_csv("SamplePredict.csv")
	v_res.to_csv("ValidationPredict.csv")

if __name__ == "__main__":
	sample,validation = importData()

	parameters = [(10,10),(10,10),(10,10),(10,10),(10,10)]

	Replicating(sample,validation,parameters)
