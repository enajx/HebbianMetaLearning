import torch
import torch.nn as nn


class MLP_heb(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space, action_space):
        super(MLP_heb, self).__init__()

        self.fc1 = nn.Linear(input_space, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, action_space, bias=False)

    def forward(self, ob):
        state = torch.as_tensor(ob[0]).float().detach()
        
        x1 = torch.tanh(self.fc1(state))   
        x2 = torch.tanh(self.fc2(x1))
        o = self.fc3(x2)  
         
        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  
    



class CNN_heb(nn.Module):
    "CNN+MLP with n=input_channels frames as input. Non-activated last layer's output"
    def __init__(self, input_channels, action_space_dim):
        super(CNN_heb, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=False)   
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)
        
        self.linear1 = nn.Linear(648, 128, bias=False) 
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.out = nn.Linear(64, action_space_dim, bias=False)
    
    
    def forward(self, ob):
        
        state = torch.as_tensor(ob.copy())
        state = state.float()
        
        x1 = self.pool(torch.tanh(self.conv1(state)))
        x2 = self.pool(torch.tanh(self.conv2(x1)))
        
        x3 = x2.view(-1)
        
        x4 = torch.tanh(self.linear1(x3))   
        x5 = torch.tanh(self.linear2(x4))
        
        o = self.out(x5)

        return x3, x4, x5, o
        # return self.pool(self.conv2(x1)).view(-1), self.linear1(x3), self.linear2(x4), o
        
