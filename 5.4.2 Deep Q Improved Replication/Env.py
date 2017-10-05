import pandas
import datetime
import numpy

class Env():
    def __init__(self,df,beta,verbose=False):
        self.action_space = [0,1,2]
        
        self.df = df
        self.beta = beta
        self.verbose = verbose

        self.current = 0
        
        self.df["res"] = self.df["Replication"] - beta*self.df["SP500"]
    
    def get_state(self):
        prev = self.df[self.current:self.current+1]
        
        sp500Ret = prev["SP500"].values[0]
        res = prev["res"].values[0]
        
        sp500Ret = numpy.exp(sp500Ret)-1
        res = numpy.exp(res)-1
        
        s_t = [sp500Ret,res]
        s_t = numpy.array(s_t)
        s_t = s_t.reshape((1,len(s_t)))
        return s_t
    
    def execute(self,action):
        reward,terminal = self.get_reward(action)
        
        return (self.get_state(),reward,terminal)
    
    def get_reward(self,action):
        if self.verbose:
            print action
        
        p1 = self.df["SP500"]
        p2 = self.df["Replication"]
        
        s1 = 0
        s2 = 0     

        #Execute Action
        if action == 1:
            #Long Portfolio
            s2 = 1
            s1 = -self.beta
        elif action == 2:
            s2 = -1
            s1 = self.beta
        elif action == 0:
            pass
        else:
            print "Invalid Action"
            raise ValueError
        
        self.current += 1
        
        #Get reward
        ret = 0
        ret += s1*p1[self.current]
        ret += s2*p2[self.current]
        
        terminal = 0
        if self.current == len(self.df)-1:
            #Current is last day, reset the encvironment
            terminal = 1
            self.current = 1
        return (ret,terminal)

if __name__ == "__main__":        
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
    
    sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
    beta = 0.958996658297
    
           
    a = Env(sample,beta,True)  
    print a.df
    print a.get_state()
    print a.execute(1) 
    print a.execute(0)

        
