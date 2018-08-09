#import cv2
#import torchvision
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import torch.onnx
from torch.autograd import Variable

from torch.utils.data import Dataset


def to_onnx():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3)

            self.pool = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(64 * 61 * 61, 1220)
            self.fc2 = nn.Linear(1220, 100)
            self.fc3 = nn.Linear(100, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = x.view(-1, 64 * 61 * 61)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE WILL BE USED: ", device)

    net = Net()
    #restore_model = 'C:\\Users\gkazanta\Desktop\model-frozen-11'
    #ckpt = torch.load(restore_model, map_location='cpu')
    #net.load_state_dict(ckpt)
    #print("Model {} restored".format(restore_model))

    dummy_input = Variable(torch.randn(1, 3, 256, 256), requires_grad=True)
    #input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(12)]
    #output_names = ["output1"]

    torch.onnx.export(net, dummy_input, "vgg_hot_dog.onnx", export_params=True, verbose=True)#, input_names=input_names, output_names=output_names)


if __name__ == "__main__":
    to_onnx()