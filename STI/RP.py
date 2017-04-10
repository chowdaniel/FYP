from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD,Adam

import pandas
import numpy
import os

def Replicating(sample,validation,N):

	s_res = pandas.DataFrame(index=sample.index[1:])
	v_res = pandas.DataFrame(index=validation.index[1:])

	s_res["^STI"] = numpy.diff(numpy.log(sample.as_matrix(columns=["^STI"])),axis=0)
	v_res["^STI"] = numpy.diff(numpy.log(validation.as_matrix(columns=["^STI"])),axis=0)

	v_error = pandas.DataFrame(index=N)
	error = []

	for n in N:
		#Import list of stocks sorted by increasing SSE
		symbols = open("Portfolio.csv","r")

		stocks = []
		for symbol in symbols:
			if symbol != "\n":
				stocks.append(symbol.replace("\n",""))
		symbols.close()
		
		#Select the stocks used for replicating portfolio
		chosen_stocks = []
		#10 Stocks with most communal info
		for i in range(n[0]):
			chosen_stocks.append(stocks[i])
		#n stocks with least communal info
		for i in range(n[1]):
			chosen_stocks.append(stocks[-i-1])
		chosen_stocks.append("^STI")


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

		#Build Deep Network
		model = Sequential()

		model.add(Dense(output_dim=4,input_dim=X.shape[1],activation="relu"))
		model.add(Dense(output_dim=2,activation="relu"))
		model.add(Dense(output_dim=1))

		opt = Adam(lr=0.001)
		model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

		model.fit(X,Y,batch_size=40,nb_epoch=100,validation_split=0,verbose=0)

		Y_pred = model.predict(X)
		s_res[n] = Y_pred

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
		v_res[n] = Y_pred

		error.append(numpy.sum(numpy.square(numpy.subtract(Y_pred,Y))))

	v_error["Error"] = error

	s_res.to_csv("SamplePredict.csv")
	v_res.to_csv("ValidationPredict.csv")
	v_error.to_csv("ValidationError.csv")

if __name__ == "__main__":

	FILENAME = "Data.csv"
	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

	sample = imported_data.loc["2014-01-01":"2015-12-31"]
	validation = imported_data.loc["2016-01-01":"2016-12-30"]

	Replicating(sample,validation,[(1,0),(1,1),(1,5),(1,10)])