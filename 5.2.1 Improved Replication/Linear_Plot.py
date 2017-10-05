import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas
import datetime
import numpy


def plot(data):
    fig,ax = plt.subplots()
    
    dates = data.index
    dates = map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),dates)    

    ax.plot(dates,numpy.cumsum(data["SP500"]),label="S&P500")
    ax.plot(dates,numpy.cumsum(data["Replication"]),label="Replication")
    try:
        ax.plot(dates,numpy.cumsum(data["Improved"]),label="Target")
    except:
        pass

    #Formatting for plot
    #Format x-axis
    months = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    fig.autofmt_xdate()
    #Label Axis
    plt.xlabel("Month/Year")
    plt.ylabel("Cumulative log Return")
    #Show legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0)
plot(sample)

validation = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0)
plot(validation)


