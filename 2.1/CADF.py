import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas
import numpy
import math
import datetime

def run_test(data):
    stock1 = "True"
    stock2 = "Predicted"
    
    df = data
    X = df[stock1]
    Y = df[stock2]
    
    start = df.index[0]
    end = df.index[-1]
   
    res = run_regression(X,Y)
    beta = res.params[1]
    print "Beta: %s" % beta
    
    df["res"] = df[stock2] - beta*df[stock1]
    
    run_cadf(df)
    run_hurst(df)      
    run_halflife(df)
    
    plot_price_series(df,start,end)    
    plot_scatter_series(df)
    plot_residuals(df,start,end)

def run_regression(X,Y):
    X = sm.add_constant(X)
    
    model = sm.OLS(Y,X)
    res = model.fit()
    
    return res

def run_cadf(df):
    print('\nCADF')
    
    cadf = ts.adfuller(df["res"])

    testStat = cadf[0]
    pValue = cadf[1]
    
    print "Test Statistic: %s" % testStat
    print "p-value: %s" % pValue

def run_hurst(df):   
    print('Hurst')
    
    gbm = numpy.log(numpy.cumsum(numpy.random.randn(100000))+1000)
    mr = numpy.log(numpy.random.randn(100000)+1000)
    tr = numpy.log(numpy.cumsum(numpy.random.randn(100000)+1)+1000)
    
    print "Hurst(GBM(Random)): %s" % hurst_calc(gbm)
    print "Hurst(MeanReverting): %s" % hurst_calc(mr)
    print "Hurst(Trending): %s" % hurst_calc(tr)
    
    ts = df["res"]
    print "Hurst: %s" % hurst_calc(ts)
    print('-----------------------------------------')
    
def hurst_calc(ts):
    lags = range(2, 100)
    
    tau = [numpy.sqrt(numpy.std(numpy.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    poly = numpy.polyfit(numpy.log(lags), numpy.log(tau), 1)
    
    return poly[0]*2.0

def run_halflife(df):
    results = df["res"].tolist()
    delta = numpy.diff(results)
    results.pop()
    
    X = results
    Y = delta

    res = run_regression(X,Y)
    beta = res.params[1]
    
    halflife = -math.log(2)/beta
    
    print "HalfLife: %s" % halflife
    print('-----------------------------------------')

def plot_price_series(df,start,end):
    fig, ax = plt.subplots()
    
    stocks = df.columns
    stock1 = stocks[0]
    stock2 = stocks[1]
    
    ax.plot(df.index, df[stock1], label=stock1)
    ax.plot(df.index, df[stock2], label=stock2)

    months = mdates.MonthLocator()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.set_xlim(start,end)
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Return' % (stock1, stock2))
    plt.legend()
    plt.show()
    
def plot_scatter_series(df):
    fig, ax = plt.subplots()
        
    stocks = df.columns
    stock1 = stocks[0]
    stock2 = stocks[1]
    
    plt.xlabel('%s Return' % stock1)
    plt.ylabel('%s Return' % stock2)
    plt.title('%s and %s Return Scatterplot' % (stock1, stock2))
    plt.scatter(df[stock1], df[stock2])
    plt.show()
    
def plot_residuals(df,start,end):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
   
    months = mdates.MonthLocator()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.set_xlim(start,end)
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.legend()
    plt.show()
     

parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")

sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
run_test(sample)

validation = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0,date_parser=parser)
run_test(validation)
