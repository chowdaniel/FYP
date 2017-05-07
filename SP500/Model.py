from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import KFold

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

	data = imported_data.loc[start1:end5]

	return data
	
def buildModel(input_dim,leaky=False):
	model = Sequential()

	leakyLayer = LeakyReLU(alpha=0.01)

	activation = "relu"
	dropout_rate = 0.5

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
	model.add(Dense(output_dim=1))

	return model

def evaluateFold(X_train,X_test,Y_train,Y_test):
	model = buildModel(X_train.shape[1],leaky=False)

	opt = Adam(lr=0.001)
	model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

	model.fit(X_train,Y_train,batch_size=40,nb_epoch=100,validation_split=0,verbose=0)

	Y_pred = model.predict(X_test)

	fold_error = numpy.sum(numpy.square(numpy.subtract(Y_pred,Y_test)))
	return fold_error

if __name__ == "__main__":
	sample = importData()

	n_folds = 5
	models = [(10,0),(10,5)]

	for model in models:
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

		#k fold cross validation
		kf = KFold(n_splits=n_folds)

		total_error = 0
		for train_index,test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			total_error += evaluateFold(X_train,X_test,Y_train,Y_test)

		print "Model: %s\tError: %f" % (model,total_error)