import torch
from torch import nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    
    def __init__(self):
        """
        Creates an instance of the SimpleNet class
        """
        super(SimpleNet, self).__init__()
                
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.fc1 = nn.Linear(2352, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        
        
    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)
    
    
    def num_flat_features(self, x):
        
        size = x.size()[1:]
        num_flat_features = 1
        for s in size:
            num_flat_features *= s
            
        return num_flat_features