from Env import Env
from collections import deque

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam

import datetime
import pandas
import numpy
import random

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
        model.add(Dense(self.action_size))
        model.add(Activation(activation))

        opt = Adam(lr=0.001)
        model.compile(optimizer=opt,loss="mse")
        return model
    
    def fit_model(self,iterations):
        epsilon = 0.2
        REPLAY_SIZE = 5000
        BATCH_SIZE = 32
        INITIAL_OBS = 300
        
        GAMMA = 0.8
        
        D = deque()
        
        s_t = self.env.get_state()

        t = -INITIAL_OBS
        
        while t < iterations:
            a_t = numpy.zeros(self.action_size)
                 
            if random.random() <= epsilon:
                #Pick Random Action
                action_index = random.randint(0,self.action_size-1)
                a_t[action_index] = 1
            else:
                #Pick Greedt Action
                q = self.model.predict(s_t)
                action_index = numpy.argmax(q)
                
                a_t[action_index] = 1
            
            #Execute action
            s_t1,r,terminal = self.env.execute(action_index)
            
            temp = (s_t,action_index,r,s_t1,terminal)
            D.append(temp)
            if len(D) > REPLAY_SIZE:
                D.popleft()

            if t > 0:
                minibatch = random.sample(D, BATCH_SIZE)
                
                inputs = []
                targets = []
                
                for i in range(BATCH_SIZE):
                    element = minibatch[i]
                    
                    s_t = element[0]
                    a_t = element[1]
                    r = element[2]
                    s_t1 = element[3]
                    terminal = element[4]
                    
                    inputs.append(s_t[0])
                    
                    target = self.model.predict(s_t)[0]
                    q = self.model.predict(s_t1)
                    
                    if terminal == 1:
                        target[a_t] = r
                    else:
                        target[a_t] = r + GAMMA * numpy.max(q)
                    targets.append(target)
                inputs = numpy.array(inputs)
                targets = numpy.array(targets)

                self.model.train_on_batch(inputs,targets)

            t += 1

if __name__ == "__main__":
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
    
    sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
    beta = 0.958996658297
    
           
    env = Env(sample,beta)        
    q = DeepQ(env)
    
    q.fit_model(100000)
        
    