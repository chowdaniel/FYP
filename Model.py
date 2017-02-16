from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils.visualize_util import plot

import pandas
import numpy
import os

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


def AutoEncoder():
	#Build and Fit Model
	model = Sequential()

	model.add(Dense(output_dim=HIDDEN_LAYER_DIM,input_dim=data_length,activation="relu"))
	model.add(Dense(output_dim=data_length,activation="relu"))

	model.compile(optimizer="sgd",loss="mse",metrics=["accuracy"])

	model.fit(data,data,batch_size=10,nb_epoch=20,validation_split=0)

	plot(model,to_file="Encoder.png",show_shapes=True)
	return
	#Calculate Error for each Symbol
	y_pred = model.predict(data)


	diff = data - y_pred
	SSE = numpy.sum(numpy.square(diff),axis=0)

	txt = ""

	for element in SSE:
		txt = txt + str(element.item())
		txt = txt + ","

	txt = txt[:-1]
	txt += "\n"

	out = open("Results.csv","a")
	out.write(txt)
	out.close()

def parseResults():
	FILENAME = "Merged.csv"

	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
	stocks = list(imported_data)

	my_data = numpy.genfromtxt("Results.csv", delimiter=',')
	s = numpy.sum(my_data,axis=0)

	res = []

	print "%d rows of data found" % len(s)

	for i in range(len(s)):
		temp = [stocks[i],s[i].item()]
		res.append(temp)

	res = sorted(res,key=lambda x:x[1])

	return res

def indexWeights(s):
	#s is the results from parseResults

	#Import data for S&P500
	stocks = ["^GSPC"]

	#Take the 10 stocks with the lowest error
	for i in range(10):
		stocks.append(s[i][0])

	for i in range(-10,0):
		stocks.append(s[i][0])

	#Build stock historical data into DataFrame
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

	model.fit(X,Y,batch_size=10,nb_epoch=20,validation_split=0)

	plot(model,to_file="Model.png",show_shapes=True)

	Y_pred = model.predict(X)

	res = []
	for i in range(len(Y)):
		temp = [Y_pred[i].item(),Y[i].item()]
		res.append(temp)

	print res

if __name__ == "__main__":
#	for i in range(10):
#		AutoEncoder()
	AutoEncoder()

	res = parseResults()
	indexWeights(res)




