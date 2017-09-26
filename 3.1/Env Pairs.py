import pandas
import datetime
import numpy

from Env import Env

class Backtest():
    def __init__(self,env,z_entry,z_stop):
        self.env = env

        action = (z_entry,z_stop)
        action_space = self.env.action_space
        self.action_index = action_space.index(action)

    def execute(self):
        terminal = 0
        
        returns = []
        while terminal == 0:
            s_t1,r,terminal = self.env.execute(self.action_index)
            
            returns += r
        return numpy.mean(returns)
        
        
if __name__ == "__main__":
    ENTRY_MAX = 1.5
    STOP_MAX = 3.0
    ACTION_STEP_SIZE = 0.1
    STATE_SIZE = 5
    
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
    
    sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
    validation = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0,date_parser=parser)
    
    beta = 0.958996658297    

    max_ret = 0
    max_entry = 0
    max_stop = 0
    
    entry = 0.1
    stop = round(entry + ACTION_STEP_SIZE,1)    
    while entry <= ENTRY_MAX:
        print entry
        while stop <= STOP_MAX:
            sample_env = Env(sample,beta) 
            validation_env = Env(validation,beta)
            
            sample_backtest = Backtest(sample_env,entry,stop)
            ret = sample_backtest.execute()
            
            if ret > max_ret:
                max_ret = ret
                max_entry = entry
                max_stop = stop
            
            validation_backtest = Backtest(validation_env,entry,stop)
            ret = validation_backtest.execute()
            
            stop = round(stop + ACTION_STEP_SIZE,1)
        entry = round(entry + ACTION_STEP_SIZE,1)
        stop = round(entry + ACTION_STEP_SIZE,1)    

    

    
    
    