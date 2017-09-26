from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

import statsmodels.api as sm

import pandas
import numpy

import matplotlib.pyplot as plt
    
def get_data():
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
    
    sampleStart = start1
    sampleEnd = end4
    validationStart = start5
    validationEnd = end6
    
    sample = imported_data.loc[sampleStart:sampleEnd]
    validation = imported_data.loc[validationStart:validationEnd]
    
    return (sample,validation)

def build_model(input_dim,leaky=False):
    model = Sequential()
    
    leakyLayer = LeakyReLU(alpha=0.01)
    
    activation = "relu"
    dropoutRate = 0.2
    
    if leaky:
        model.add(Dense(15,input_dim=input_dim))
        model.add(leakyLayer)
    else:
        model.add(Dense(15,input_dim=input_dim,activation=activation))
    model.add(Dropout(dropoutRate))
    
    if leaky:
        model.add(Dense(10))
        model.add(leakyLayer)
    else:
        model.add(Dense(10,activation=activation))
    model.add(Dropout(dropoutRate))
    
    model.add(Dense(1))
    
    return model

def RP(nLow,nHigh):
    nFits = 20
    
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
      
    sample,validation = get_data()

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
    
    #Begin Prediction for models
    prediction = pandas.DataFrame(index=valData.index[1:])
    for i in range(nFits):
        model = build_model(nLow+nHigh,leaky=False)

        opt = Adam(lr=0.001)
        model.compile(optimizer=opt,loss="mse",metrics=["accuracy"])
        
        earlyStop = EarlyStopping(monitor="val_loss",min_delta=0,patience=10,mode="min")

        history = model.fit(trainX,trainY,batch_size=40,epochs=200,validation_data=(valX,valY),callbacks=[earlyStop],verbose=0)

        Y_pred = model.predict(valX)
        prediction[i] = Y_pred

    prediction["Mean"] = numpy.mean(prediction,axis=1)
    prediction["True"] = valY
    prediction["Diff"] = prediction["Mean"] - prediction["True"]
    prediction.to_csv("output.csv")    

    SSE = numpy.sum(numpy.square(prediction["Diff"]))
    
    print "ReLU Model: %f" % SSE

def Linear(nLow,nHigh):
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
      
    sample,validation = get_data()

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
    print numpy.exp(trainY)
    threshold = 0.03
    for i in range(len(trainY)):
        if trainY[i] < -threshold:
            trainY[i] = threshold

    trainX = sm.add_constant(trainX)
    model = sm.OLS(trainY,trainX)
    results = model.fit()    
    
    valX = sm.add_constant(valX)
    Y_pred = results.predict(valX)
    
    table = pandas.DataFrame(data=valY)
    table["True"] = valY
    table["Pred"] = Y_pred
    table["Diff"] = table["Pred"] - table["True"]

    SSE = numpy.sum(numpy.square(table["Diff"]))  
    print "Linear Model: %f" % SSE
    
    table = pandas.DataFrame(index=trainData.index[1:])
    table["True"] = trainY
    table["Predicted"] = results.predict(trainX)
    
    table.to_csv("Linear_Sample.csv")
    
    table = pandas.DataFrame(index=valData.index[1:])
    table["True"] = valY
    table["Predicted"] = Y_pred
    
    table.to_csv("Linear_Validation.csv")

if __name__ == "__main__":
    n_low = 40
    n_high = 10
    
    #RP(n_low,n_high)
    Linear(n_low,n_high)





