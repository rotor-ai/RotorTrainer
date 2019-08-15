import torch
from torch import nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    
    def __init__(self, num_labels):
        """
        Creates an instance of the SimpleNet class. 
        SimpleNet requires an input of size 64 x 64
        :param num_labels: The final number of predicted labels
        """
        
        super(SimpleNet, self).__init__()

        num_channels_1 = 3
        num_channels_2 = 6
        num_channels_3 = 12

        full_size = 2352  # Must be tweaked depending on pooling values and convolutions
        downsize_1 = 120
        downsize_2 = 84
        
        self.conv1 = nn.Conv2d(num_channels_1, num_channels_2, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(num_channels_2, num_channels_3, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.fc1 = nn.Linear(full_size, downsize_1)
        self.fc2 = nn.Linear(downsize_1, downsize_2)
        self.fc3 = nn.Linear(downsize_2, num_labels)
        
        
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