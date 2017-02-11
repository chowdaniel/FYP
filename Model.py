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

data_length = data.shape[1]

model.add(Dense(output_dim=HIDDEN_LAYER_DIM,input_dim=data_length,activation="relu"))
model.add(Dense(output_dim=data_length,activation="relu"))

model.compile(optimizer="sgd",loss="mse",metrics=["accuracy"])

model.fit(data,data,batch_size=50,nb_epoch=10)
