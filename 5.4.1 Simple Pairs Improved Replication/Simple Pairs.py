import pandas
import numpy
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Pairs():
    def __init__(self,df,beta):
        self.n_days = 3
        
        self.df = df
        self.beta = beta
        
        self.stock1 = "SP500"
        self.stock2 = "Replication"
        
        self.df["res"] = df[self.stock2] - self.beta*df[self.stock1]
        
        self.generate_stats()
        
    def generate_stats(self):
        mean = numpy.zeros(len(self.df["res"]))
        sd = numpy.zeros(len(self.df["res"]))
        
        res = self.df["res"]
        for i in range(self.n_days,len(res)+1):
            sl = res.iloc[i-self.n_days:i]
            mean[i-1] = numpy.mean(sl)
            sd[i-1] = numpy.std(sl)
        
        self.df["Mean"] = mean
        self.df["SD"] = sd
        
    def run_test(self,entry,stop):
        p1 = self.df["SP500"]
        p2 = self.df["Replication"]
        
        s1 = 0
        s2 = 0

        res = self.df["res"].tolist()
        mean = self.df["Mean"].tolist()
        sd = self.df["SD"].tolist()  
        date = self.df.index
        
        returns = []
        dates = []
        position = 0
        
        counter = self.n_days-1
        while counter < len(self.df):
            z = (res[counter]-mean[counter])/sd[counter]

            ret = 0
            ret += s1*p1[counter]
            ret += s2*p2[counter] 
            returns.append(ret)
            dates.append(date[counter])

            #Check stop loss condition
            if position == 1:
                if z > stop:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
            elif position == -1:
                if z < -stop:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
                    
            #Check closing condition
            if position == 1:
                if z < 0:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
            elif position == -1:
                if z > 0:
                    s1 = 0
                    s2 = 0
                    
                    position = 0   

            if z > entry and z < stop:
                s2 = -1
                s1 = self.beta

                position = 1
            elif z < -entry and z > -stop:
                s2 = 1
                s1 = -self.beta
       
                position = -1
                
            counter += 1
            
        output = pandas.DataFrame(index=dates)
        output["returns"] = returns
        
        return output
    
def plot_returns(returns):
    dates = returns.index
    ts = returns["returns"]

    fig,ax = plt.subplots()

    ax.plot(dates,numpy.cumsum(ts))

    #Formatting for plot
    #Format x-axis
    months = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    fig.autofmt_xdate()
    #Label Axis
    plt.xlabel("Month/Year")
    plt.ylabel("Cumulative log Return")
        
if __name__ == "__main__":
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")

    sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
    validation = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0,date_parser=parser)
    
    beta = 0.636983293243
    
    a = Pairs(sample,beta)
    b = Pairs(validation,beta)
    
    ENTRY_MAX = 1.5
    STOP_MAX = 3.0
    ACTION_STEP_SIZE = 0.1
    
    max_ret = 0
    max_entry = 0
    max_stop = 0
    
    test = 0
    if test:
        entry = 0.1
        stop = round(entry + ACTION_STEP_SIZE,1)    
        while entry <= ENTRY_MAX:
            while stop <= STOP_MAX:
                ret = a.run_test(entry,stop)
                ret = numpy.mean(ret["returns"])
                
                if ret > max_ret:
                    max_ret = ret
                    max_entry = entry
                    max_stop = stop
                    
                stop = round(stop + ACTION_STEP_SIZE,1)
            entry = round(entry + ACTION_STEP_SIZE,1)
            stop = round(entry + ACTION_STEP_SIZE,1) 
        print max_ret
        print max_entry
        print max_stop

    entry = 0.2
    stop = 1.3
    
    sample_returns = a.run_test(entry,stop)
    validation_returns = b.run_test(entry,stop)
    
    plot_returns(sample_returns)
    plot_returns(validation_returns)
    
    
    
    
    
    
    
    