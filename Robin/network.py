import torch.nn as nn
import torch
import torch.nn.functional as F
class ResNet(nn.Module):
    def __init__(self,dim, m,o):
        super(ResNet, self).__init__()
        self.Ix = torch.zeros([dim,m]).cuda()
        self.Ix[0,0] = 1
        self.Ix[1,1] = 1
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)

        
        self.outlayer = nn.Linear(m, o)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = F.relu(y)**deg
        y = self.fc2(y)
        y = F.relu(y)**deg
        y = y+s
        
        s=y
        y = self.fc3(y)
        y = F.relu(y)**deg
        y = self.fc4(y)
        y = F.relu(y)**deg
        y = y+s
        
        
        output = self.outlayer(y)
        return output
deg = 2