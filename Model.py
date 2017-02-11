from keras.models import Sequential
from keras.layers import Dense,Activation

import pandas
import numpy
import os

#Model Parameters
FILENAME = "Merged.csv"
HIDDEN_LAYER_DIM = 200


#Begin Code
model = Sequential()

imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
data = imported_data.as_matrix()

data = numpy.log(data)
#data = numpy.diff(data,axis=0)

#print data

data_length = data.shape[1]

#Build and Fit Model
model.add(Dense(output_dim=HIDDEN_LAYER_DIM,input_dim=data_length,activation="relu"))
model.add(Dense(output_dim=data_length,activation="relu"))

model.compile(optimizer="sgd",loss="mse",metrics=["accuracy"])

model.fit(data,data,batch_size=10,nb_epoch=20,validation_split=0)

#Calculate Error for each Symbol
y_pred = model.predict(data)


diff = data - y_pred
SSE = numpy.sum(numpy.square(diff),axis=0)

txt = ""

for element in SSE:
	txt = txt + str(element.item())
	txt = txt + ","

txt += "\n"

out = open("Results.txt","a")
out.write(txt)
out.close()
