## Author: Xinghua Qu (quxinghua17@gmail.com)
import warnings
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import loadtxt

# load data from files. 
# Original data files are recorded by Leon on game HappyBird
# Future data can be from any game in Intellik if they are collected in a readonable way.
# Ask Leon for details of collecting data

class DataLoad():
    def __init__(self, data_dir, name):
        super(DataLoad, self).__init__()
        self.data_dir  = data_dir
        self.name = name

    def load(self):
        data = loadtxt("{}/{}".format(self.data_dir, self.name), delimiter=",", unpack=False)
#         with open("{}/{}".format(self.data_dir, self.name), "rt") as f:
#             data = f.readlines()
        self.data = data
        self.data_list = data
        return data
    
    
#     def str2list(self):
#         data_list = []
#         for line in self.data:
#             line = line[:-1]
#             float_line = [int(d) for d in line if d is not ',']
#             data_list.append(float_line)
#         self.data_list = data_list
#         return data_list
    
    def state_action_split(self):
        States = []
        Actions= []
        for data in self.data_list:
            States.append(data[:-1])
            Actions.append(int(data[-1]))
        self.States  = States
        self.Actions = Actions
        return States, Actions
    
    def train_test_split(self, train_size):
        # training_percentage: the percentage of the data will be used for training (the rest is for testing)
        # random_state here is for reproducbility
        X_train, X_test, y_train, y_test = train_test_split(self.States, self.Actions, train_size= train_size, random_state=1)
        return X_train, X_test, y_train, y_test
        
    
if __name__== "__main__":
    loader    = DataLoad('data_set','test.txt')
    data      = loader.load()
    print(data)
#     data_list = loader.str2list()
    States, Actions = loader.state_action_split()
    print(States, Actions)
    X_train, X_test, y_train, y_test = loader.train_test_split(0.8)