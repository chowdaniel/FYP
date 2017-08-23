from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

import pandas
import numpy

import matplotlib.pyplot as plt

def get_data(k):
	start1 = "2014-01-02"
	start2 = "2014-06-02"
	start3 = "2015-01-02"
	start4 = "2015-06-01"
	start5 = "2016-01-04"
	start6 = "2016-06-01"

	end1 = "2014-05-30"
	end2 = "2014-12-31"
	end3 = "2015-05-29"
	end4 = "2015-12-31"
	end5 = "2016-05-31"
	end6 = "2016-12-30"

	FILENAME = "Data.csv"
	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

	sampleEnd = [end1,end2,end3,end4,end5]
	validationStart = [start2,start3,start4,start5,start6]
	validationEnd = [end2,end3,end4,end5,end6]

	sample = imported_data.loc[start1:sampleEnd[k]]
	validation = imported_data.loc[validationStart[k]:validationEnd[k]]

	return (sample,validation)

def build_model(input_dim,leaky=False):
	model = Sequential()

	leakyLayer = LeakyReLU(alpha=0.01)

	activation = "relu"
	dropoutRate = 0.2

	if leaky:
		model.add(Dense(6,input_dim=input_dim))
		model.add(leakyLayer)
	else:
		model.add(Dense(6,input_dim=input_dim,activation=activation))
	model.add(Dropout(dropoutRate))

	if leaky:
		model.add(Dense(4))
		model.add(leakyLayer)
	else:
		model.add(Dense(4,activation=activation))
	model.add(Dropout(dropoutRate))
	model.add(Dense(1))

	return model

def RP(nLow,nHigh):
	symbols = open("Portfolio.csv","r")

	stocks = []
	for symbol in symbols:
		if symbol != "\n":
			stocks.append(symbol.replace("\n",""))
	symbols.close()

	#Select the stocks used for replicating portfolio
	chosen_stocks = []
	#Stocks with most communal info
	for i in range(nLow):
		chosen_stocks.append(stocks[i])
	#Stocks with least communal info
	for i in range(nHigh):
		chosen_stocks.append(stocks[-i-1])
	chosen_stocks.append("^GSPC")

	mse = []
	for k in range(5):
		sample,validation = get_data(k)

		s_res = pandas.DataFrame(index=sample.index[1:])
		v_res = pandas.DataFrame(index=validation.index[1:])

		s_res["^GSPC"] = numpy.diff(numpy.log(sample.as_matrix(columns=["^GSPC"])),axis=0)
		v_res["^GSPC"] = numpy.diff(numpy.log(validation.as_matrix(columns=["^GSPC"])),axis=0)

		#Setup Training and Validation Set
		trainData = pandas.DataFrame()
		valData = pandas.DataFrame()

		for stock in chosen_stocks:
			prices = sample[stock]
			trainData[stock] = prices

			prices = validation[stock]
			valData[stock] = prices

		#Extract X and Y as numpy arrays
		trainX = trainData.as_matrix(columns=chosen_stocks[0:-1])
		trainY = trainData.as_matrix(columns=chosen_stocks[-1:])
		valX = valData.as_matrix(columns=chosen_stocks[0:-1])
		valY = valData.as_matrix(columns=chosen_stocks[-1:])

		#log prices
		trainX = numpy.log(trainX)
		trainY = numpy.log(trainY)
		valX = numpy.log(valX)
		valY = numpy.log(valY)

		#log returns
		trainX = numpy.diff(trainX,axis=0)
		trainY = numpy.diff(trainY,axis=0)
		valX = numpy.diff(valX,axis=0)
		valY = numpy.diff(valY,axis=0)

		#Build and fit Deep Network
		model = build_model(nLow+nHigh,leaky=False)

		opt = Adam(lr=0.001)
		model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

		earlyStopping = EarlyStopping(monitor="val_loss",min_delta=0,patience=3,mode="min")

		history = model.fit(trainX,trainY,batch_size=40,epochs=100,validation_data=(valX,valY),callbacks=[earlyStopping],verbose=0)

		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.show()
		
		Y_pred = model.predict(trainX)
		#s_res[counter] = Y_pred
		Y_pred = model.predict(valX)
		#v_res[counter] = Y_pred

		diff = valY - Y_pred
		MSE = numpy.sum(numpy.square(diff),axis=0)/valX.shape[0]

		mse.append(MSE)
	print mse

if __name__ == "__main__":
	RP(5,5)