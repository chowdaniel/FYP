from keras.models import Sequential
from keras.layers import Dense,Activation

import pandas
import numpy
import os

#Model Parameters
FILENAME = "Merged.csv"

#Begin Code
imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

data = imported_data.as_matrix()

#data = numpy.log(data)
#data = numpy.diff(data,axis=0)

data_length = data.shape[1]

#print data

def AutoEncoder(encoder_activation,decoder_activation,hidden_dim):

	#Build and Fit Model
	model = Sequential()

	model.add(Dense(output_dim=hidden_dim,input_dim=data_length,activation=encoder_activation))
	model.add(Dense(output_dim=data_length,activation=decoder_activation))

	model.compile(optimizer="sgd",loss="mse",metrics=["accuracy"])

	model.fit(data,data,batch_size=10,nb_epoch=10,validation_split=0,verbose=2)


	#Calculate Error for each Symbol
	y_pred = model.predict(data)

	diff = data - y_pred
	SSE = numpy.sum(numpy.square(diff),axis=0)

	return SSE


if __name__ == "__main__":
	activation = ["softmax","softplus","softsign","relu","tanh","sigmoid","hard_sigmoid","linear"]

	ENCODER_ACTIVATION = activation[3]
	DECODER_ACTIVATION = activation[3]

	HIDDEN_DIM = 200
	N_RUNS = 20


	#Write headers on results file
	results = open("Results.csv","w")
	stocks = list(imported_data)

	for stock in stocks:
		results.write(stock)
		results.write(",")
	results.write("\n")

	#Fit Autoencoder multiple times and record results into results file
	for i in range(N_RUNS):
		res = AutoEncoder(ENCODER_ACTIVATION,DECODER_ACTIVATION,HIDDEN_DIM)

		for sse in res:
			results.write(str(sse.item()))
			results.write(",")
		results.write("\n")

	results.close()

