from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

import statsmodels.api as sm

import pandas
import numpy
   
def get_data():
    FILENAME = "Data.csv"
    imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)
    
    sampleStart = "2014-01-02"
    sampleEnd = "2015-12-31"
    validationStart = "2016-01-04"
    validationEnd = "2016-12-30"
    
    sample = imported_data.loc[sampleStart:sampleEnd]
    validation = imported_data.loc[validationStart:validationEnd]
    
    return (sample,validation)

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

    #Replacing returns that are below threshold
    trainY2 = numpy.zeros(trainY.shape)
    threshold = 0.02
    for i in range(len(trainY)):
        if numpy.exp(trainY[i]) < 1-threshold:
            trainY2[i] = numpy.log(1+threshold)
        else:
            trainY2[i] = trainY[i]

    trainX = sm.add_constant(trainX)
    model = sm.OLS(trainY2,trainX)
    results = model.fit()    
    
    valX = sm.add_constant(valX)
    Y_pred = results.predict(valX)
    
    table = pandas.DataFrame(data=valY)
    table["True"] = valY
    table["Pred"] = Y_pred
    table["Diff"] = table["Pred"] - table["True"]

    SSE = numpy.sum(numpy.square(table["Diff"]))  
    print "Linear Model SSE: %f" % SSE
    
    #Output Replicating Portfolio returns on sample and validation set
    table = pandas.DataFrame(index=trainData.index[1:])
    table["SP500"] = trainY
    table["Replication"] = results.predict(trainX)
    
    table.to_csv("Linear_Sample.csv")
    
    table = pandas.DataFrame(index=valData.index[1:])
    table["SP500"] = valY
    table["Replication"] = Y_pred
    
    table.to_csv("Linear_Validation.csv")

if __name__ == "__main__":
    n_low = 40
    n_high = 10

    Linear(n_low,n_high)





