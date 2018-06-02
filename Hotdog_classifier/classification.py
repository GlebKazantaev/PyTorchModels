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

from torch.utils.data import Dataset



# fig = plt.figure()
# for i in range(len(hot_dog_dataset_train)):
#     sample = hot_dog_dataset_train[i]
#     plt.imshow(sample['image'])
#     plt.pause(1.1)
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class HotDogsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, train, transform=None):
        self.root_dir = root_dir #+ ("/train" if train else "/test")
        self.pref = "train" if train else "test"
        self.transform = transform

        self.hot_dog_path = self.root_dir + "/{}/hotdog/".format(self.pref)
        self.not_hot_dog_path = self.root_dir + "/{}/not_hotdog/".format(self.pref)

        self.dataset_len = len(os.listdir(self.hot_dog_path)) + len(os.listdir(self.not_hot_dog_path))

        # Load data set
        self.images = []
        self.labels = {}
        for x in os.listdir(self.hot_dog_path):
            self.images.append(self.hot_dog_path + x)
            self.labels.update({self.hot_dog_path + x:1})

        for x in os.listdir(self.not_hot_dog_path):
            self.images.append(self.not_hot_dog_path + x)
            self.labels.update({self.not_hot_dog_path + x: 0})

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img_name = os.path.join(self.images[idx])
        image = io.imread(img_name)
        #print(img_name, " ", image.shape)
        label = np.array([float(self.labels[img_name])])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        #print(image.shape, " ", label)
        return {'image': torch.from_numpy(image.transpose((2, 0, 1))), 'label': label}


def test(dataset_root_dir: str, restore_model: str):
    """ """

    """
        LOAD TRAIN & TEST DATASETS 
    """
    hot_dog_dataset_train = HotDogsDataset(train=True, root_dir=dataset_root_dir, transform=transforms.Compose([
                                               Rescale((256,256)),
                                               ToTensor()
                                           ]))
    train_dataloader = DataLoader(hot_dog_dataset_train, batch_size=4, shuffle=True, num_workers=1)

    hot_dog_dataset_test = HotDogsDataset(train=False, root_dir=dataset_root_dir, transform=transforms.Compose([
                                               Rescale((256,256)),
                                               ToTensor()
                                           ]))
    test_dataloader = DataLoader(hot_dog_dataset_test, batch_size=4, shuffle=True, num_workers=4)

    """
        SETUP CNN MODEL
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE WILL BE USED: ", device)

    classes = ('not hotdog', 'hotdog')

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

    net = Net()
    net = net.to(device)

    if restore_model is not None:
        net.load_state_dict(torch.load(restore_model))
        print("Model {} restored".format(restore_model))

    """
        SETUP LOSS FUNCTION
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    """
        START TRAINING
    """
    last_accuracy = None
    epoch = 0
    #for epoch in range(1):  # loop over the dataset multiple times
    while True:
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs
            inputs, labels = data['image'].float(), data['label']
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long().view(4))
            loss.backward()
            optimizer.step()

            # print statistics
            print('.', end='')
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        epoch += 1

        for dl, type in zip([test_dataloader, train_dataloader], ['test', 'train']):
            correct = 0
            total = 0
            with torch.no_grad():
                for data in dl:
                    images, labels = data['image'].float(), data['label'].long().view(4)
                    images, labels = images.to(device), labels.to(device)

                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            cur_accuracy = (100 * correct / total)
            print('Accuracy of the network on the ' + type + ' set with ' + str(total) + ' test images: %d %%' % cur_accuracy)

        torch.save(net.state_dict(), './model-frozen-{}'.format(epoch))

    print('Finished Training')

    """
        PREDICT RESULTS
    """
    #dataiter = iter(test_dataloader)
    #data = dataiter.next()
    #images, labels = data['image'].float(), data['label'].long().view(4)
    # print images
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #outputs = net(images)
    #_, predicted = torch.max(outputs, 1)
    #print('Predicted: ', ' '.join('%5s, ' % classes[predicted[j]] for j in range(4)))
    #plt.pause(1e9)


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Train data-set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Test data-set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 5)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.conv3 = nn.Conv2d(32, 64, 3)

            self.pool = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(64 *  10 * 10, 220)
            self.fc2 = nn.Linear(220, 100)
            self.fc3 = nn.Linear(100, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(-1, 64 * 10 * 10)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


    # Predict
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    restore_model = sys.argv[2] if len(sys.argv) > 2 else None
    test(sys.argv[1], restore_model)
    #main()