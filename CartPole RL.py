#!/usr/bin/env python
# coding: utf-8

# # Deep Reinforcement Algorithm in OpenAI gym environment
# 
# We shall build a deep neural network and use RL to solve a cart and pole balancing problem

# In[1]:


import sys
print(sys.version)


# In git bash, we type the following commands:
# 
# 
# git clone https://github.com/openai/gym
# 
# cd gym
# 
# pip install -e . # minimal install
# 
# 
# 

# This downloads the bare minimums for the OpenAI Gym environment. 

# In[2]:


import gym
print(gym.__version__)

import keras
print(keras.__version__)


# If it does not show 'Using Theano backend' and instead shows "Using Tensorflow backend" or anything else;
# go to .keras folder in the directory where Anaconda is installed;
# open the 'keras' JSON file in a text editor and change whatever is written in the section marked as "backend" to "Theano"

# In[3]:


import random
import math
import numpy as np
from collections import deque


# ## Setting up OpenAI Gym environment

# In[4]:


env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    
    for t in range(100):
        env.render()
        
        print(observation)
        
        action = env.action_space.sample()
        
        observation, reward, done, info = env.step(action)
        
        if done:
            break


# In[5]:


print(env.action_space)
print(env.observation_space)


# In[6]:


print(env.observation_space.high)
print(env.observation_space.low)


# ## Defining parameters

# In[7]:


# training parameters

n_episodes = 1000    # no. of episodes
n_win_ticks = 195    # every time step is a tick(in OpenAI); done state = win_tick
max_env_steps = None # for OPen AI

# RL parameters

gamma = 1.0          # Discount factor: measure of how far ahead in time the algorithm looks
                     # might not be good now 
                     # To prioritise rewards in the distant future, the value is kept one 
                     # deciding whether or not we want to value current rewards or future rewards   
epsilon = 1.0        # exploration factor starting from one   
                     # Exploration : Choose a uniformly random choice, random force to use; agent choosing an alg thinking it 
                     # will have the best long term effect
                     # avoid local minimum
                     # exploitation: when you keep doing what you were doing; exploration: when you try something new   
epsilon_min = 0.01   # starting with high expl with and then immediately start lowering this 
epsilon_decay = 0.995 # how quickly it will stop exploring
alpha = 0.01         # Learning rate: how big you take a leap in finding optimal policy
                     # it will determine to what extent new info will override old info
                     # alpha=0 means no learning; alpha = 1 means considering only recent info   
alpha_decay = 0.01   # lowering alpha

batch_size = 64      # 64 samples
monitor = False      # stuff for OpenAI
quiet = False        # control print statements 


# Environment Parameters

# for AI Gym

memory = deque(maxlen = 100000)    # custom list parameter, setting(controlling) max length 
env = gym.make("CartPole-v0")
if max_env_steps is not None: 
    env.max_episode_steps = max_env_steps


# ## Building the neural network

# In[8]:


# building the neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#Model definition

model = Sequential()
model.add(Dense(24, input_dim=4, activation = 'relu'))
                # 24 neurons, input dimensions = 4 as current environment has 4 paramters
                # activation is rectified linear unit

#adding hidden layers

model.add(Dense(48, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))  
                # we have force to the left and to the right
                # so two possible outputs; so 2 neurons

#how to compile this
model.compile(loss = 'mse', optimizer = Adam(lr = alpha, decay = alpha_decay))    # learning rate is alpha


# ## Defining necessary functions

# In[9]:


# defining necessary functions

#setting up memory
def remember(state, action, reward, next_state, done):          # reward that we got, checking whether it is done ot not
    memory.append((state, action, reward, next_state, done))
    
#choose action: pick what to do    
def choose_action(state, epsilon):
    return env.action.sample() if (np.random.random() <= epsilon) else np.argmax(model.predict(state))
                                                #if no. chosen randomly from action space <= 1(at start)
                                                #if not, we shall get our model making up prediction based off current state
                                                            #i.e., for exploration stage, prediction on force and direction
        
def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0-math.log10((t+1)*epsilon_decay)))
                                                #towards the end we'd be decreasing substantially
                                                # in the beginning, right up at epsilon
        
# getting preprocess
def preprocess_state(state):
    return np.reshape(state, [1, 4])            # transposing state matrix to a column

#going through replay
def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))
    
    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma + np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])
        
    #fit our model
    #using the actions to train our model
    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
                                                                    #verbose: whther or not to make print statements outof this
        
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


# In[10]:


# define run function
# training our model which would choose the best action to do


def run():
    scores = deque(maxlen = 100)
    
    for e in range(n_episodes):
        state = preprocess_state(env.reset())    # start from the beginning each and everytime 
        done = False
        i = 0                                    # time-set = 0
        
        while not done:                          # while done is false
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            env.render()                         # rendering so that we can see what's goin' on
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
        
        scores.append(i)
        
        mean_score = np.mean(scores)
        
        if mean_score >= n_win_ticks and e >= 100:
            if not quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e-100))
            return e-100
        if e % 20 == 0 and not quiet:
            print('[episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
            
        
        replay(batch_size,epsilon)
        
    if not quiet: print('did not solve after {} episodes'.format(e))
    return e


# ## Training the network

# In[ ]:



# copying and pasting all the things from above

# as running the environment already initiated is not a good idea






import gym
import keras
import random
import math
import numpy as np
from collections import deque



# training parameters

n_episodes = 1000    # no. of episodes
n_win_ticks = 195    # every time step is a tick(in OpenAI); done state = win_tick
max_env_steps = None # for OPen AI

# RL parameters

gamma = 1.0          # Discount factor: measure of how far ahead in time the algorithm looks
                     # might not be good now 
                     # To prioritise rewards in the distant future, the value is kept one 
                     # deciding whether or not we want to value current rewards or future rewards   
epsilon = 1.0        # exploration factor starting from one   
                     # Exploration : Choose a uniformly random choice, random force to use; agent choosing an alg thinking it 
                     # will have the best long term effect
                     # avoid local minimum
                     # exploitation: when you keep doing what you were doing; exploration: when you try something new   
epsilon_min = 0.01   # starting with high expl with and then immediately start lowering this 
epsilon_decay = 0.995 # how quickly it will stop exploring
alpha = 0.01         # Learning rate: how big you take a leap in finding optimal policy
                     # it will determine to what extent new info will override old info
                     # alpha=0 means no learning; alpha = 1 means considering only recent info   
alpha_decay = 0.01   # lowering alpha

batch_size = 64      # 64 samples
monitor = False      # stuff for OpenAI
quiet = False        # control print statements 


# Environment Parameters

# for AI Gym

memory = deque(maxlen = 100000)    # custom list parameter, setting(controlling) max length 
env = gym.make("CartPole-v0")
if max_env_steps is not None: 
    env.max_episode_steps = max_env_steps
    
    
# building the neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#Model definition

model = Sequential()
model.add(Dense(24, input_dim=4, activation = 'relu'))
                # 24 neurons, input dimensions = 4 as current environment has 4 paramters
                # activation is rectified linear unit

#adding hidden layers

model.add(Dense(48, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))  
                # we have force to the left and to the right
                # so two possible outputs; so 2 neurons

#how to compile this
model.compile(loss = 'mse', optimizer = Adam(lr = alpha, decay = alpha_decay))    # learning rate is alpha


# defining necessary functions

#setting up memory
def remember(state, action, reward, next_state, done):          # reward that we got, checking whether it is done ot not
    memory.append((state, action, reward, next_state, done))
    
#choose action: pick what to do    
def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(model.predict(state))
                                                #if no. chosen randomly from action space <= 1(at start)
                                                #if not, we shall get our model making up prediction based off current state
                                                            #i.e., for exploration stage, prediction on force and direction
        
def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0-math.log10((t+1)*epsilon_decay)))
                                                #towards the end we'd be decreasing substantially
                                                # in the beginning, right up at epsilon
        
# getting preprocess
def preprocess_state(state):
    return np.reshape(state, [1, 4])            # transposing state matrix to a column

#going through replay
def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))
    
    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma + np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])
        
    #fit our model
    #using the actions to train our model
    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
                                                                    #verbose: whther or not to make print statements outof this
        
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
        
# define run function
# training our model which would choose the best action to do


def run():
    scores = deque(maxlen = 100)
    
    for e in range(n_episodes):
        state = preprocess_state(env.reset())    # start from the beginning each and everytime 
        done = False
        i = 0                                    # time-set = 0
        
        while not done:                          # while done is false
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            env.render()                         # rendering so that we can see what's goin' on
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
        
        scores.append(i)
        
        mean_score = np.mean(scores)
        
        if mean_score >= n_win_ticks and e >= 100:
            if not quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e-100))
            return e-100
        if e % 20 == 0 and not quiet:
            print('[episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
            
        
        replay(batch_size,epsilon)
        
    if not quiet: print('did not solve after {} episodes'.format(e))
    return e





run()


