import torch.nn as nn
import torch.nn.functional as F
import torch

# Reference: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch

class VGG_16(nn.Module):
    # Defines the network parameters
    def __init__(self, numClasses = 10):
        super(VGG_16, self).__init__()
        # Conv2d (Input Channels, Output Channels, Kernel Size, Stride, Padding)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        # Dropout Filters
        self.dropout1 = nn.Dropout2d(0.5)
        # Fully Connected Layers (Input Size, Output Size)
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, numClasses)

    # Define the sequence of the of the layers
    def forward(self, x):
        x = self.conv1(x)                   # 1x of First Section
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)                   # 2x of First Section
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)                   # 1x of Second Section
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv7(x)                   # 2x of Second Section
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv8(x)                   # 3x of Second Section
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Changes the shape of the output of the convolution layer to fit the input of the fully connected layer
        x = x.view(-1, 512 * 1 * 1)

        x = self.fc1(x)                     # Fully Connected Layer 1
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)                     # Fully Connected Layer 2
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)                     # Fully Connected Layer 3
        output = F.log_softmax(x, dim=1)
        return output
