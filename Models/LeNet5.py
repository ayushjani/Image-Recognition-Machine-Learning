import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet5(nn.Module):
    # Defines the network parameters
    def __init__(self, numClasses = 10):
        super(LeNet5, self).__init__()

        # parameters of convolution layer = (input channals, output channels, kernel size, stride, padding)
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 0)

        # Fully connected layer: input size*image size, output size
        self.fc1 = nn.Linear(64*6*6, 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84, numClasses)

    # Define the sequence of the of the layers
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        # Changes the shape of the output of the convolution layer to fit the input of the fully connected layer
        x = x.view(-1, 64*6*6)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output