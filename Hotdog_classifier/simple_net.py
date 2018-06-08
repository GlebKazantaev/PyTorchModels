import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import HotDogsDataset, Rescale, ToTensor
from torchvision import transforms


def simple_net(dataset_root_dir: str, restore_model: str):
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

            self.fc1 = nn.Linear(64 * 61 * 61, 100)
            self.fc2 = nn.Linear(100, 100)
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

    if len(restore_model) > 0:
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
                    total += labels.size(0)
                    for id, prediction in enumerate(outputs.data):
                        res = torch.nn.functional.softmax(prediction, dim=0)
                        _, rid = torch.max(res, 0)
                        if rid == labels[id]:
                            correct += 1

                    #correct += (predicted == labels).sum().item()
            cur_accuracy = (100. * correct / float(total))
            print('Accuracy of the network on the ' + type + ' set with ' + str(total) + ' test images: %f %%' % cur_accuracy)

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