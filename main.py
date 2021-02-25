## Author: Xinghua Qu (quxinghua17@gmail.com)
## Date: 24 Feb 2021
import torch.nn as nn
import torch.nn.functional as F
import warnings
from data_loader import DataLoad
from model import Intellik_Net
import argparse
from argparse import ArgumentParser
import torch.optim as optim
import torch

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default = 4, help='The maximum step for training')
    parser.add_argument('--layer_num', type=int, default = 2, help='The maximum step for training')
    parser.add_argument('--hidden_layer_dims', type=list, default = [20,20], help='The maximum step for training')
    parser.add_argument('--out_dim', type=int, default = 2, help='The maximum step for training')
    parser.add_argument('--max_epoch', type=int, default = 10000, help='The maximum step for training')
    parser.add_argument('--learning_rate', type=float, default = 1e-4, help='The maximum step for training')
    args = parser.parse_args()
    return args

def main(args):
    # create model
    input_dim, layer_num, hidden_layer_dim_list, out_dim = 4, 2, [20, 40], 2
    model = Intellik_Net(input_dim, layer_num, hidden_layer_dim_list, out_dim)
    
    # load dataset
    loader    = DataLoad('data_set','test.txt')
    data      = loader.load()
    States, Actions = loader.state_action_split()
    X_train, X_test, y_train, y_test = loader.train_test_split(0.8)
    
    tensor_X_train = torch.FloatTensor(X_train)  # transform to torch tensor
    tensor_y_train = torch.LongTensor(y_train)
    
    tensor_X_test = torch.FloatTensor(X_test)  # transform to torch tensor
    tensor_y_test = torch.LongTensor(y_test)
    
    # Create train_loader
    train_data = []
    for i in range(len(X_train)):
        train_data.append([tensor_X_train[i], tensor_y_train[i]])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
    
    # Create test_loader
    test_data = []
    for i in range(len(X_test)):
        test_data.append([tensor_X_test[i], tensor_y_test[i]])
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=32)
    
    # Training setting
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    ###### Start training ######################################
    max_step  = args.max_epoch
    for epoch in range(max_step):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_num += 1
        #### calculate the accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("\rEpoch: {:d}/{:d}  epoch_loss: {:.4f} Test accuracy: {:.2f}% ".format(epoch+1, max_step, running_loss/batch_num, 100 * correct / total, end='\n'))
    return model        

if __name__== "__main__":
    args = args()
    model = main(args)