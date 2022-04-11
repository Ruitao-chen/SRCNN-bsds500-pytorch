import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out=F.relu(self.conv3(out))
        out=F.relu(self.conv4(out))
        out = self.conv5(out)

        return out

if __name__ == '__main__':
    a = torch.randn(32, 1, 5, 5)
    net = SRCNN()
    print(net(a).shape)