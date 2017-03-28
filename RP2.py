from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD,Adam

import pandas
import numpy
import os

def Replicating(L1,L2,L3,n):

	#Import list of stocks sorted by increasing SSE
	symbols = open("Portfolio.txt","r")

	stocks = []
	for symbol in symbols:
		if symbol != "\n":
			stocks.append(symbol.replace("\n",""))
	symbols.close()
	
	#Select the stocks used for replicating portfolio
	chosen_stocks = []
	#10 Stocks with most communal info
	for i in range(10):
		chosen_stocks.append(stocks[i])
	#n stocks with least communal info
	for i in range(n):
		chosen_stocks.append(stocks[-i-1])
	chosen_stocks.append("^GSPC")


#Calibration Phase====================================================

	#Import data for stocks
	data = pandas.DataFrame()

	for stock in chosen_stocks:
		path = os.path.join("Data",stock + ".csv")
	
		d = pandas.read_csv(path,header=0,index_col=0)
		d.columns = [stock]

		data = data.merge(d,how="outer",left_index=True,right_index=True)

	#Extract X and Y as numpy arrays
	X = data.as_matrix(columns=chosen_stocks[0:-1])
	Y = data.as_matrix(columns=chosen_stocks[-1:])

	#log prices
	X = numpy.log(X)
	Y = numpy.log(Y)

	#log returns
	X = numpy.diff(X,axis=0)
	Y = numpy.diff(Y,axis=0)

	#Iterate through all elements in dataset
	for row_index in range(len(X)):
		for col_index in range(len(X[0])):
			
			#Replace if below threshold level
			if X[row_index][col_index] < -0.01:
				X[row_index][col_index] = 0.01


	#Build Deep Network
	model = Sequential()

	model.add(Dense(output_dim=4,input_dim=X.shape[1]))
	model.add(Dense(output_dim=2))
	model.add(Dense(output_dim=1))

	opt = Adam(lr=0.001)

	model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

	model.fit(X,Y,batch_size=20,nb_epoch=50,validation_split=0,verbose=2)

	Y_pred = model.predict(X)

	output = open("Calibration2Results" + str(n) + ".csv","w")
	output.write("Date,SnP Return,Portfolio Return\n")

	dates = data.index

	for i in range(len(Y)):
		output.write(dates[i+1])
		output.write(",")
		output.write(str(Y[i].item()))
		output.write(",")
		output.write(str(Y_pred[i].item()))
		output.write("\n")
		
	output.close()


#Start of Validation Phase====================================================

	#Import data for stocks
	data = pandas.DataFrame()

	for stock in chosen_stocks:
		path = os.path.join("Validation",stock + ".csv")
	
		d = pandas.read_csv(path,header=0,index_col=0)
		d.columns = [stock]

		data = data.merge(d,how="outer",left_index=True,right_index=True)

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

	output = open("Validation2Results" + str(n) + ".csv","w")
	output.write("Date,SnP Return,Portfolio Return\n")

	dates = data.index

	for i in range(len(Y)):
		output.write(dates[i+1])
		output.write(",")
		output.write(str(Y[i].item()))
		output.write(",")
		output.write(str(Y_pred[i].item()))
		output.write("\n")
		
	output.close()


if __name__ == "__main__":
	activation = ["softmax","softplus","softsign","relu","tanh","sigmoid","hard_sigmoid","linear"]

	L1 = activation[3]
	L2 = activation[3]
	L3 = activation[3]

	Replicating(L1,L2,L3,40)