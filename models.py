import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # input image size = 320 x 192 x 1
        self.conv1 = nn.Conv2d(1, 6, 5) # --> 316 * 188 * 6
        self.pool = nn.MaxPool2d(2, 2) # --> 158 * 94 * 6
        self.conv2 = nn.Conv2d(6, 16, 5) # --> 154 * 90 * 16 --> 77 * 45 * 16
        self.fc1 = nn.Linear(16 * 77 * 45, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 77 * 45)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x