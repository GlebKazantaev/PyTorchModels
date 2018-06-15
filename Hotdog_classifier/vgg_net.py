import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import HotDogsDataset, HotDogsDatasetEval, Rescale, ToTensor
from common import imshow

class VggNet(nn.Module):
    def __init__(self, model):
        super(VggNet, self).__init__()
        self.vgg_model = model
        self.vgg_model.classifier._modules['6'] = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.vgg_model(x)
        return x


def vgg_train(dataset_root_dir: str, restore_model: str, dump_to_onnx: str):
    """ """

    """
        LOAD TRAIN & TEST DATASETS 
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    hot_dog_dataset_train = HotDogsDataset(train=True, root_dir=dataset_root_dir, transform=transforms.Compose([
        Rescale((224, 224)), #normalize,
        ToTensor(),
    ]))
    train_dataloader = DataLoader(hot_dog_dataset_train, batch_size=4, shuffle=True, num_workers=1)

    hot_dog_dataset_test = HotDogsDataset(train=False, root_dir=dataset_root_dir, transform=transforms.Compose([
        Rescale((224, 224)), #normalize,
        ToTensor(),
    ]))
    test_dataloader = DataLoader(hot_dog_dataset_test, batch_size=4, shuffle=True, num_workers=4)

    """
        SETUP CNN MODEL
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE WILL BE USED: ", device)

    classes = ('not hotdog', 'hotdog')

    net = VggNet(torchvision.models.vgg16_bn(True))
    for param in list(net.parameters())[:-2]:
        param.requiers_grad = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    # if restore_model is not None:
    #     net.load_state_dict(torch.load(restore_model, map_location={'cuda:0': 'cpu'}))
    #     print("Model {} restored".format(restore_model))

    print(net)

    if len(dump_to_onnx) > 0:
        from torch.autograd import Variable
        dummy_input = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
        torch.onnx.export(net, dummy_input, "{}.onnx".format(dump_to_onnx), export_params=True, verbose=True)
        print("Saved ONNX model as {}.onnx".format(dump_to_onnx))
        return

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
    # for epoch in range(1):  # loop over the dataset multiple times
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

            print("Outside: input size", inputs.size(), "output_size", outputs.size())

            loss = criterion(outputs, labels.long().view(4))
            loss.backward()
            optimizer.step()

            # print statistics
            print('.', end='')
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
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

            cur_accuracy = (100. * correct / float(total))
            print('Accuracy of the network on the ' + type + ' set with ' + str(
                total) + ' test images: %f %%' % cur_accuracy)

        torch.save(net.state_dict(), './model-frozen-{}'.format(epoch))

    print('Finished Training')


def vgg_eval_model(dataset_root_dir, restore_model: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE WILL BE USED: ", device)

    net = VggNet(torchvision.models.vgg16_bn(True))
    net = net.to(device)

    classes = ('not hotdog', 'hotdog')

    if restore_model is not None and len(restore_model) > 0:
        # original saved file with DataParallel
        state_dict = torch.load(restore_model, map_location={'cuda:0': 'cpu'})
        net.load_state_dict(state_dict)
        print("Model {} restored".format(restore_model))
    else:
        print("ERROR: no restore model file found!")
        return

    #from torch.autograd import Variable
    #dummy_input = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
    # input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(12)]
    # output_names = ["output1"]
    #torch.onnx.export(net, dummy_input, "vgg_hot_dog.onnx", export_params=True, verbose=True)
    #print("SUCCESS")

    hot_dog_dataset_test = HotDogsDatasetEval(root_dir=dataset_root_dir, transform=transforms.Compose([
        Rescale((224, 224)), #normalize,
        ToTensor(),
    ]))
    test_dataloader = DataLoader(hot_dog_dataset_test, batch_size=4, shuffle=True, num_workers=4)


    for dl, type in zip([test_dataloader], ['test']):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dl:
                images, names = data['image'].float(), data['name']
                images = images.to(device)

                outputs = net(images)
                total += len(names)
                for id, prediction in enumerate(outputs.data):
                    res = torch.nn.functional.softmax(prediction, dim=0)
                    _, rid = torch.max(res, 0)
                    print('{} is {}'.format(names[id], classes[rid]))
                    imshow(torchvision.utils.make_grid(images[id]))

        # cur_accuracy = (100. * correct / float(total))
        # print('Accuracy of the network on the ' + type + ' set with ' + str(
        #     total) + ' test images: %f %%' % cur_accuracy)