from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy

model = Sequential()
model.add(Dense(1,input_dim=784,activation="sigmoid"))
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

data = numpy.random.random((1000,784))
labels = numpy.random.randint(2,size=(1000,1))

model.fit(data,labels,nb_epoch=10,batch_size=32)

