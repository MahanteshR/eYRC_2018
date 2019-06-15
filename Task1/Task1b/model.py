import torch.nn as nn
import torch.nn.functional as F

class FNet(nn.Module):
    # Extra TODO: Comment the code with docstrings
    """Fruit Net

    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*22*22, 500)
        self.fc2 = nn.Linear(500, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 5)
        pass

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        pass

if __name__ == "__main__":
    net = FNet()
