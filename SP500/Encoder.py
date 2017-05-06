from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD,Adam
import keras

import pandas
import numpy
import os

#Model Parameters
FILENAME = "Data.csv"

#Begin Code
imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
del imported_data["^GSPC"]

data = imported_data.as_matrix()

data = numpy.log(data)
data = numpy.diff(data,axis=0)
data = numpy.absolute(data)

data_length = data.shape[1]

def AutoEncoder(encoder_activation,decoder_activation,hidden_dim):

	#Build and Fit Model
	model = Sequential()

	leaky = False

	if not leaky:
		inputLayer = Dense(output_dim=hidden_dim,input_dim=data_length,activation=encoder_activation)
		hiddenLayer = Dense(output_dim=data_length,activation=decoder_activation)
		
		model.add(inputLayer)
		model.add(hiddenLayer)
	else:

		inputLayer = Dense(output_dim=hidden_dim,input_dim=data_length)
		hiddenLayer = Dense(output_dim=data_length)

		leakyLayer = keras.layers.advanced_activations.LeakyReLU(alpha=0.01)

		model.add(inputLayer)
		model.add(leakyLayer)
		model.add(hiddenLayer)
		model.add(leakyLayer)

	opt = Adam(lr=0.001)

	model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])

	model.fit(data,data,batch_size=5,nb_epoch=30,validation_split=0,verbose=2)

	#Calculate Error for each Sywmbol
	y_pred = model.predict(data)

	print y_pred

	diff = data - y_pred
	SSE = numpy.sum(numpy.square(diff),axis=0)

	return SSE


if __name__ == "__main__":
	activation = ["softmax","softplus","softsign","relu","tanh","sigmoid","hard_sigmoid","linear"]

	ENCODER_ACTIVATION = activation[3]
	DECODER_ACTIVATION = activation[3]

	HIDDEN_DIM = 20
	N_RUNS = 20

	stocks = list(imported_data)

	results = numpy.empty([N_RUNS,len(stocks)])

	#Fit Autoencoder multiple times and record results into results file
	for i in range(N_RUNS):
		print "Run %d" % (i)

		res = AutoEncoder(ENCODER_ACTIVATION,DECODER_ACTIVATION,HIDDEN_DIM)

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