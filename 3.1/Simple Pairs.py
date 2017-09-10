import pandas
import datetime
import numpy

import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Backtest():
    def __init__(self,df,z_entry,z_stop,beta=None):
        self.df = df
        self.entry = z_entry
        self.stop = z_stop
        
        self.stock1 = "True"
        self.stock2 = "Predicted"
        
        self.n_days = 5
        
        X = df[self.stock1]
        Y = df[self.stock2]
        res = self.run_regression(X,Y)
        
        self.beta = 0
        if beta == None:
            self.beta = res.params[1]
        else:
            self.beta = beta
        #print "Beta: %s" % self.beta
        
        self.df["res"] = df[self.stock2] - self.beta*df[self.stock1]
        
        self.generate_stats()
        
        start = df.index[0]
        end = df.index[-1]
        #self.plot_residuals(start,end)
        
    def run_regression(self,X,Y):
        X = sm.add_constant(X)
        
        model = sm.OLS(Y,X)
        res = model.fit()
        
        return res

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
        
    def plot_residuals(self,start,end):
        fig, ax = plt.subplots()
        ax.plot(self.df.index, self.df["res"], label="Residuals")
        
        ax.plot(self.df.index,self.df["Mean"],label="Mean")
        ax.plot(self.df.index,self.df["Mean"]+self.df["SD"],label="1Z")
        
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
        
    def start_backtest(self):
        ret = 0
        returns = []
        position = 0
        
        p1 = self.df[self.stock1]
        p2 = self.df[self.stock2]
        
        s1 = 0
        s2 = 0

        res = self.df["res"].tolist()
        mean = self.df["Mean"].tolist()
        sd = self.df["SD"].tolist()
    
        for i in range(len(res)):
            #Calculate Z-score
            z = 0
            try:
                z = (res[i]-mean[i])/sd[i]
            except:
                continue
            #print z
            
            ret = 0
            ret += s1*p1[i]
            ret += s2*p2[i] 
            returns.append(ret)
            
            #Check stop loss condition
            if position == 1:
                if z > self.stop:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
            elif position == -1:
                if z < -self.stop:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
            
            #Check closing condition
            if position == 1:
                if z < 0:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
                    #print "Close"
            elif position == -1:
                if z > 0:
                    s1 = 0
                    s2 = 0
                    
                    position = 0                   
                    #print "Close"
                    
            #Check opening condition
            if position == 0:
                if z > self.entry and z < self.stop:
                    s2 = -1
                    s1 = self.beta

                    position = 1
                    #print "1"
                elif z < -self.entry and z > -self.stop:
                    s2 = 1
                    s1 = -self.beta
           
                    position = -1
                    #print "-1"
                    
        #print returns
        return returns

parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")

sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
validation = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0,date_parser=parser)

z_entry = 0.5
z_stop = 2

b = Backtest(sample,z_entry,z_stop)
b.start_backtest()

entry = 0.1
stop = entry+0.05

max_stat = 0
max_entry = 0
max_stop = 0

while entry <= 2:
    #print entry
    while stop <=3:
        b = Backtest(sample,entry,stop)
        ret = b.start_backtest()
        print ret
        stat = numpy.mean(ret)/numpy.std(ret)
        
        if stat > max_stat:
            max_stat = stat
            max_entry = entry
            max_stop = stop
        
        stop += 0.05
    entry += 0.05
    stop = entry + 0.05
    
    print "Max Stat: %f\tEntry: %f\tStop: %f" % (max_stat,max_entry,max_stop)

