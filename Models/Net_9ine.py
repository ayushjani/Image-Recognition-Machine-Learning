import torch.nn as nn
import torch.nn.functional as F
import torch

class Net_9ine(nn.Module):
    # Defines the network parameters
    def __init__(self, numClasses = 10):
        super(Net_9ine, self).__init__()

        # Define Spatial Separable Convolution
        self.conv1a = nn.Conv2d(3, 64, kernel_size=(5, 1), stride=1, padding=2)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=(1, 5), stride=1, padding=2)
        self.conv2a = nn.Conv2d(64, 128, kernel_size=(5, 1), stride=1, padding=0)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=(1, 5), stride=1, padding=0)
        # Define Depthwise Separable Convolution
        self.depthwise1 = nn.Conv2d(128, 128, kernel_size=3)
        self.pointwise1 = nn.Conv2d(128, 256, kernel_size=1)
        # Dropout Layer
        self.dropout1 = nn.Dropout2d()
        # Normalisation Batch
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        # Fully connected layer: input size*image size, output size
        self.fc1 = nn.Linear(256*1*1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, numClasses)

    # Define the sequence of the of the layers
    def forward(self, x):
        x = self.conv1a(x)              # Spatial Separable Convolution 1
        x = self.conv1b(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2a(x)              # Spatial Separable Convolution 2
        x = self.conv2b(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        x = self.depthwise1(x)           # Depthwise Convolution 1
        x = self.bn1(x)
        x = F.relu(x)
        x = self.depthwise1(x)
        x = self.bn1(x)
        x = F.avg_pool2d(x, 2)

        x = self.pointwise1(x)           # Pointwise Convolution 1
        x = self.bn2(x)
        x = F.relu(x)

        # Forcefully changes the dimensions of the output of Pointwise
        # Convolution to fit the input of Fully Connected Layer 1
        x = x.view(-1, 256*1*1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output