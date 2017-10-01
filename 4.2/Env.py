import pandas
import datetime
import numpy

ENTRY_MAX = 1.5
STOP_MAX = 3.0
ACTION_STEP_SIZE = 0.1
STATE_SIZE = 5

class Env():
    def __init__(self,df,beta,verbose=False):
        self.action_space = []
        
        self.df = df
        self.beta = beta
        self.verbose = verbose

        self.current = STATE_SIZE-1
        self.n_days = 3
        
        self.df["res"] = self.df["Predicted"] - beta*self.df["True"]
        
        self.initialize_action_space()
        self.generate_stats()
        
    def initialize_action_space(self):
        entry = 0.1
        stop = round(entry + ACTION_STEP_SIZE,1)
        
        while entry <= ENTRY_MAX:
            while stop <= STOP_MAX:
                element = (entry,stop)
                self.action_space.append(element)
                
                stop = round(stop + ACTION_STEP_SIZE,1)
            entry = round(entry + ACTION_STEP_SIZE,1)
            stop = round(entry + ACTION_STEP_SIZE,1)
            
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
    
    def get_state(self):
        s_t = self.df[self.current-STATE_SIZE+1:self.current+1]["res"].tolist()
        s_t = numpy.reshape(numpy.array(s_t),(1,STATE_SIZE))
        
        return s_t
    
    def execute(self,actionIndex):
        action = self.action_space[actionIndex]
        
        reward,terminal = self.get_reward(action)
        
        return (self.get_state(),reward,terminal)
    
    def get_reward(self,action):
        if self.verbose:
            print action
        entry = action[0]
        stop = action[1]
        
        p1 = self.df["True"]
        p2 = self.df["Predicted"]
        
        s1 = 0
        s2 = 0

        res = self.df["res"].tolist()
        mean = self.df["Mean"].tolist()
        sd = self.df["SD"].tolist()  
        
        returns = []
        position = 0
        #Find Opening
        while self.current < len(self.df) and position == 0:
            z = 0
            try:
                z = (res[self.current]-mean[self.current])/sd[self.current]
            except:
                continue
            
            if z > entry and z < stop:
                s2 = -1
                s1 = self.beta

                position = 1
            elif z < -entry and z > -stop:
                s2 = 1
                s1 = -self.beta
       
                position = -1
            else:
                returns.append(0)                
            
            if self.verbose:
                print "State: %s\tz-score: %f\tPosition: %d" % (self.get_state(),z,position)
            self.current += 1
            
        #Find Closing            
        while self.current < len(self.df) and position != 0:
            z = 0
            z = (res[self.current]-mean[self.current])/sd[self.current]

            ret = 0
            ret += s1*p1[self.current]
            ret += s2*p2[self.current] 
            returns.append(ret)

            #Check stop loss condition
            trade = ""
            if position == 1:
                if z > stop:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
                    trade = "Stop"
            elif position == -1:
                if z < -stop:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
                    trade = "Stop"
                    
            #Check closing condition
            if position == 1:
                if z < 0:
                    s1 = 0
                    s2 = 0
                    
                    position = 0
                    trade = "Close"
            elif position == -1:
                if z > 0:
                    s1 = 0
                    s2 = 0
                    
                    position = 0                   
                    trade = "Close" 
                    
            if self.verbose:
                print "State: %s\tz-score: %f\tPosition: %d\tTrade: %s" % (self.get_state(),z,position,trade)
            self.current += 1

        if self.verbose:
            print returns
        reward = returns
        terminal = 0
        if self.current == len(self.df):
            terminal = 1
            self.current = STATE_SIZE-1
        
        return (reward,terminal)
        

        
if __name__ == "__main__":        
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
    
    sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
    beta = 0.958996658297
    
           
    a = Env(sample,beta,True)      
    print a.execute(1) 
    print ""
    print a.execute(30) 