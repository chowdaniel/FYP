from keras.models import Sequential
from keras.layers import Dense,Activation

import pandas
import numpy
import os

def Replicating(L1,L2,L3):

	#Add S&P500 to list of data to import
	stocks = ["^GSPC"]

	symbols = open("Portfolio.txt","r")

	for symbol in symbols:
		if symbol != "\n":
			stocks.append(symbol.replace("\n",""))
	symbols.close()


	#Import data for stocks
	data = pandas.DataFrame()

	for stock in stocks:
		path = os.path.join("Data",stock + ".csv")
	
		d = pandas.read_csv(path,header=0,index_col=0)
		d.columns = [stock]

		data = data.merge(d,how="outer",left_index=True,right_index=True)


	#Extract X and Y as numpy arrays
	X = data.as_matrix(columns=stocks[1:])
	Y = data.as_matrix(columns=stocks[0:1])

	#log prices
	X = numpy.log(X)
	Y = numpy.log(Y)

	#log returns
	X = numpy.diff(X,axis=0)
	Y = numpy.diff(Y,axis=0)

	#Build Deep Network
	model = Sequential()

	model.add(Dense(output_dim=4,input_dim=X.shape[1]))
	model.add(Dense(output_dim=2))
	model.add(Dense(output_dim=1))

	model.compile(optimizer="sgd",loss="mse",metrics=["accuracy"])

	model.fit(X,Y,batch_size=10,nb_epoch=10,validation_split=0)

	Y_pred = model.predict(X)

	res = []
	for i in range(len(Y)):
		temp = [Y_pred[i].item(),Y[i].item()]
		res.append(temp)

	print res

if __name__ == "__main__":
	activation = ["softmax","softplus","softsign","relu","tanh","sigmoid","hard_sigmoid","linear"]

	L1 = activation[-1]
	L2 = activation[-1]
	L3 = activation[-1]

	Replicating(L1,L2,L3)