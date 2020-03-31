import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3D(nn.Module):
    def __init__(self, cin, cout):
        super(Conv3D, self).__init__()
        self.conv1 = nn.Conv2d(cin, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(8, cout, 4, 3, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class FC(nn.Module):
    def __init__(self, cin, cout):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(cin, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 40)
        self.fc4 = nn.Linear(40, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


 