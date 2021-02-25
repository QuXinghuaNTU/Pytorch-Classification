## Author: Xinghua Qu (quxinghua17@gmail.com)
import torch.nn as nn
import torch.nn.functional as F
import warnings

# This file is to create the neural network models

# parameter settings: 
# input_dim ---> dimension of the observations collected from IntelliK
# out_dim ---> dimension of the actions collected from IntelliK 
# layer_num  ---> how many hidden layers you want for your network
# hidden_layer_dim_list ---> The list that show the dimension of your each hidden dimension

class Intellik_Net(nn.Module):
    def __init__(self, input_dim, layer_num, hidden_layer_dim_list, out_dim):
        super(Intellik_Net, self).__init__()
        if layer_num<1: warnings.warn("The input layer number must be not less than 1!")
        if layer_num != len(hidden_layer_dim_list): warnings.warn("The input layer number does not match the number of hidden layers given!")
        self.fc_in  = nn.Linear(input_dim, hidden_layer_dim_list[0])
        self.hidden = nn.ModuleList()
        for k in range(layer_num - 1):
            self.hidden.append(nn.Linear(hidden_layer_dim_list[k], hidden_layer_dim_list[k+1]))
        self.fc_out = nn.Linear(hidden_layer_dim_list[-1], out_dim)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return x

