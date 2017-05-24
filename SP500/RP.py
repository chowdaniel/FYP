from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.initializers import Constant
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

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

	data = imported_data.loc[start1:end6]

	return data

def buildModel(input_dim,leaky=False):
	model = Sequential()

	bias_init = Constant(0.1)
	leakyLayer = LeakyReLU(alpha=0.01)

	activation = "relu"
	dropout_rate = 0.5

	if leaky:
		model.add(Dense(4,input_dim=input_dim))
		model.add(leakyLayer)
	else:
		model.add(Dense(4,input_dim=input_dim,activation=activation,bias_initializer=bias_init))
	model.add(Dropout(dropout_rate))

	if leaky:
		model.add(Dense(2))
		model.add(leakyLayer)
	else:
		model.add(Dense(2,activation=activation,bias_initializer=bias_init))
	model.add(Dense(1))

	return model

def evaluateModel(X_train,X_test,Y_train,Y_test):
	coeff = 0

	plot = True
	learning_rate = 0.001

	while numpy.absolute(coeff) < 5:
		model = buildModel(X_train.shape[1],leaky=False)

		opt = Adam(lr=0.001)
		model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

		history = model.fit(X_train,Y_train,batch_size=40,epochs=100,validation_data=(X_test,Y_test),verbose=0)

		Y_pred = model.predict(X_test)

		coeff = numpy.std(Y_pred)/numpy.mean(Y_pred)
		print "Model Fitted with coeff: %f" % (coeff)
	#print numpy.transpose(Y_pred)
	#print "Coefficient: %f" % (coeff)

	MSE = numpy.sum(numpy.square(numpy.subtract(Y_pred,Y_test)))/len(Y_test)
	print MSE
	fig,ax = plt.subplots()

	if plot:
		ax.plot(history.history["loss"],label="Sample Loss")
		ax.plot(history.history["val_loss"],label="Validation Loss")

		plt.title("Model Loss")
		plt.xlabel("epoch")
		plt.ylabel("Loss")
		ax.legend(loc="upper left")
		plt.show()

if __name__ == "__main__":
	sample = importData()
	n_obs = sample.shape[0]

	n_split = 5
	split_size = int(n_obs/n_split)
	models = [(10,0),(10,5),(10,10),(10,15),(10,20),(10,25),(10,30)]
	models = [(10,0)]

	for model in models:
		print "Building Model (%d,%d)" % (model[0],model[1])
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
		for i in range(model[0]):
			chosen_stocks.append(stocks[i])
		#Stocks with least communal info
		for i in range(model[1]):
			chosen_stocks.append(stocks[-i-1])
		chosen_stocks.append("^GSPC")

		#Import data for chosen stocks
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

		for i in range(1,n_split):
			print "\nSample Split: %d-%d\t Validation Split: %d" % (1,i,i+1)
			train_index = list(range(0,i*split_size))
			test_index = list(range(i*split_size,(i+1)*split_size))

			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			evaluateModel(X_train,X_test,Y_train,Y_test)