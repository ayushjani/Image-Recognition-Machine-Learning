from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Models.Alexnet import AlexNet
from Models.VGG16 import VGG_16
from Models.LeNet5 import LeNet5
from Models.Net_9ine import Net_9ine
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import sklearn
from sklearn import metrics

#LR for VGG16 = 0.0001(77% acc)
#LR for AlexNet = 0.0005 (61% Acc)
#LR for Lenet and Mod_lenet = 0.001 (70% acc)


# Function that will be invoked to train the selected model with the specified training dataset
def train(log_interval, model, device, trainloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    iteration_track = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Variables which are used to develop the evaluation functionality
        running_loss += loss
        iteration_track = iteration_track + 1

        # Prints the loss at the end of an EPOCH
        if batch_idx * len(data) == len(trainloader.dataset) - len(data):
            print('\nTraining model... Epoch #{}: Loss = {} '.format(epoch + 1, loss.item()))

    training_Loss.append(running_loss / iteration_track)

# Function that will be invoked to test the model with the specified testing dataset
def test(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_loss = test_loss * 1000

    # Variables which are used to develop the evaluation functionality
    testing_Loss.append(test_loss)
    testing_Acc.append(correct / len(testloader.dataset))

    # Prints the accuracy at the end of each epoch
    print('Testing... Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

# Function to calculate the required tables and graphs
def Evaluation(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in testloader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            y_true.append(target.tolist())
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            temp_pred = pred
            temp_pred = temp_pred.view(1000)
            y_pred.append(temp_pred.tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()

    target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    labels = ['plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Flattens the list of list
    y_true_flattened = [y for x in y_true for y in x]
    y_pred_flattened = [y for x in y_pred for y in x]

    # Prints the confusion matrix
    print(metrics.confusion_matrix(y_true_flattened, y_pred_flattened, target), '\n')
    print(metrics.classification_report(y_true_flattened, y_pred_flattened, target))

    # Prints the loss and accuracy graphs
    plt.plot(training_Loss, label='Training loss')
    plt.plot(testing_Loss, label='Testing loss')
    plt.title(' Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.plot(testing_Acc, label='Testing Accuracy')
    plt.title(' Model Accuracy ')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# Empty lists used to store loss and accuracy values
training_Loss = []
testing_Loss = []
testing_Acc = []

def main():
    num_epoches = 1
    log_interval = 10
    torch.manual_seed(1)
    training_batch_size = 200
    testing_batch_batch_size = 1000

    save_model = False

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    ######################   Torchvision    ###########################
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transformed = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10('../data', train=True,
                                            download=True, transform=transformed)
    trainloader = torch.utils.data.DataLoader(trainset, training_batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10('../data', train=False,
                                           download=True, transform=transformed)
    testloader = torch.utils.data.DataLoader(testset, testing_batch_batch_size,
                                             shuffle=False, num_workers=0)

    model = LeNet5().to(device)
    learnRate = 0.001

    # Defines the optimizer
    optimizer = optim.Adam(model.parameters(), learnRate)

    # Starts timing
    since = time.time()

    # loops over the dataset specified number of times
    for epoch in range(num_epoches):
        # Invokes the training function for the model
        train(log_interval, model, device, trainloader, optimizer, epoch)

        # Prints the total runtime after final training
        if epoch == num_epoches - 1:
            print('Finished Training')
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Invokes the testing function for the model
        test(model, device, testloader)

        # Carries out the Evaluation after final epoch is complete
        if epoch == num_epoches - 1:
            Evaluation(model, device, testloader)

    if save_model:
        torch.save(model.state_dict(), "./results/model.pt")
        print('\nModel saved as model.pt')


if __name__ == '__main__':
    main()