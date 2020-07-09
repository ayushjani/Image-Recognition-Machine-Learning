#ALexNet code written by Ayush
import torch.nn as nn
import torch.nn.functional as F
import torch


# https://www.learnopencv.com/understanding-alexnet/

class AlexNet(nn.Module):
    # Defines the network parameters
    def __init__(self, numClasses=10):
        super(AlexNet, self).__init__()

        # parameters of convolution layer = (input channels, output channels, kernel size, stride, padding)
        self.conv1 = nn.Conv2d(3, 64, 11, 4, 2)
        self.conv2 = nn.Conv2d(64, 192, 5, 1, 2)
        self.conv3 = nn.Conv2d(192, 384, 3, 1, 2)
        self.conv4 = nn.Conv2d(384, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)

        # It will 'filter' out some of the input by the probability
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout2d()

        # Fully connected layer: input size*image size, output size
        self.fc1 = nn.Linear(256 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, numClasses)

    # Define the sequence of the of the layers
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        # Changes the shape of the output of the convolution layer to fit the input of the fully connected layer
        x = x.view(-1, 256 * 1 * 1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return x
