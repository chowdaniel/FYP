from Env import Env

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam

import datetime
import pandas
import numpy

class DeepQ():
    def __init__(self,env):      
        self.env = env
        self.action_size = len(env.action_space)
        
        self.model = self.build_model()
        
    def build_model(self):
        activation = 'relu'
        
        model = Sequential()
        
        model.add(Dense(self.action_size/2,input_dim=5))
        model.add(Activation(activation))
        model.add(Dense(self.action_size/2))
        model.add(Activation(activation))
        
        model.add(Dense(self.action_size))

        model.load_weights("model.h5")

        opt = Adam(lr=0.0001)
        model.compile(optimizer=opt,loss="mse")
        return model
    
    def test_model(self):
        s_t = self.env.get_state()
        terminal = 0
        
        returns = []
        while terminal == 0:
            #Find and take greedy action
            q = self.model.predict(s_t)
            action_index = numpy.argmax(q)
            
            #Execute action
            s_t1,r,terminal = self.env.execute(action_index)

            returns += r
                
            s_t = s_t1
        print numpy.mean(returns)

if __name__ == "__main__":
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
    
    sample = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0,date_parser=parser)
    beta = 0.958996658297
    
           
    env = Env(sample,beta)        
    q = DeepQ(env)
    
    q.test_model()