from Env import Env
from collections import deque

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam

import datetime
import pandas
import numpy
import random

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class DeepQ():
    def __init__(self,env):      
        self.env = env
        self.action_size = len(env.action_space)
        
        self.model = self.build_model()
        self.actions = []
        
    def build_model(self):
        activation = 'tanh'
        
        model = Sequential()
        
        model.add(Dense(3,input_dim=2))
        model.add(Activation(activation))
        model.add(Dense(2))
        model.add(Activation(activation))
        
        model.add(Dense(3))

        model.load_weights("model.h5")

        opt = Adam(lr=1e-6)
        model.compile(optimizer=opt,loss="mse")
        return model
    
    def fit_model(self,iterations):
        epsilon = 0.1
        REPLAY_SIZE = 2000
        BATCH_SIZE = 32
        INITIAL_OBS = 100
        
        GAMMA = 0.99
        
        D = deque()
        
        s_t = self.env.get_state()

        t = -INITIAL_OBS
        
        while t < iterations:
            if random.random() <= epsilon:
                #Pick Random Action
                action = random.randint(0,self.action_size-1)
            else:
                #Pick Greedy Action
                s_t = numpy.array(s_t)
                
                q = self.model.predict(s_t)
            
                action = numpy.argmax(q)
    
            #Execute action
            s_t1,r,terminal = self.env.execute(action)
            r = r * 1000

            temp = (s_t,action,r,s_t1,terminal)
        
            D.append(temp)
            if len(D) > REPLAY_SIZE:
                D.popleft()

            if t >= 0:
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
                    if terminal == 1:
                        target[a_t] = r
                    else:
                        q = self.model.predict(s_t1)
                        
                        target[a_t] = r + GAMMA * numpy.max(q)
                    targets.append(target)

                inputs = numpy.array(inputs)
                targets = numpy.array(targets)

                loss = self.model.train_on_batch(inputs,targets)
                
                if numpy.isnan(loss):
                    print "Iteration %d: Loss - %f" % (t,loss)
                    print minibatch
                    print q
                    return
                if t%1000 == 0:
                    print "Iteration %d: Loss - %f" % (t,loss)

            t += 1
            s_t = s_t1
        print q    
        self.model.save_weights("model.h5",overwrite=True)
    
    def test_model(self):
        s_t = self.env.get_state()
        terminal = 0
        
        date = self.env.df.index
        
        returns = []
        dates = []
        while terminal == 0:
            #Find and take greedy action
            q = self.model.predict(s_t)

            action_index = numpy.argmax(q)
            self.actions.append(action_index)
            #Execute action
            dates.append(date[self.env.current])
            s_t1,r,terminal = self.env.execute(action_index)

            returns.append(r)
            
                
            s_t = s_t1

        self.convert_index()
        self.plot(dates,returns)
        
    def convert_index(self):
        for i in range(len(self.actions)):
            self.actions[i] = env.action_space[self.actions[i]]
        
        print self.actions
    
    def plot(self,dates,ts):
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
        #Show legend
      

if __name__ == "__main__":
    parser = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
    
    sample = pandas.read_csv("Linear_Sample.csv",header=0,index_col=0,date_parser=parser)
    validation = pandas.read_csv("Linear_Validation.csv",header=0,index_col=0,date_parser=parser)
    beta = 0.636983293243
    
    fit = 1
    sample_test = 1       
    validation_test = 1
    
    if fit:
        env = Env(sample,beta)        
        q = DeepQ(env)
        q.fit_model(10000)
    if sample_test:
        env = Env(sample,beta)
        q = DeepQ(env)
        q.test_model()
    if validation_test:
        env = Env(validation,beta)
        q = DeepQ(env)
        q.test_model()       
        
        
        
    