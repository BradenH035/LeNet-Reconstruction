# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
   

    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        relu = nn.ReLU()
        # certain definitions
        self.shape = input_shape
        self.cv1 = nn.Conv2d(3, 6, 5, 1)
        self.rl1 = relu
        self.mp1 = nn.MaxPool2d(2, 2)

        self.cv2 = nn.Conv2d(6, 16, 5, 1)
        self.rl2 = relu
        self.mp2 = nn.MaxPool2d(2, 2)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(16*5*5, 256)
        self.rl3 = relu

        self.fc2 = nn.Linear(256, 128)
        self.rl4 = relu
        
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        shapeDict = {}
        # certain operations
        out = self.cv1(x)
        out = self.rl1(out)
        out = self.mp1(out)
        shapeDict[1] = out.shape

        out = self.cv2(out)
        out = self.rl2(out)
        out = self.mp2(out)
        shapeDict[2] = out.shape

        out = self.flat(out)
        shapeDict[3] = out.shape

        out = self.fc1(out)
        out = self.rl3(out)
        shapeDict[4] = out.shape

        out = self.fc2(out)
        out = self.rl4(out)
        shapeDict[5] = out.shape

        out = self.fc3(out)
        shapeDict[6] = out.shape

        return out, shapeDict


def count_model_params():
    model = LeNet()
    model_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    adjustedParams = model_params / 1000000 # number of trainable parameters (in millions)
    return adjustedParams


def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
