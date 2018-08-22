import time
import torch
import torch.nn as nn
import platform
from common import Logger, logging
from torch.utils.data import DataLoader
from torchvision import transforms

from DeblurResNet.dataset import DeblurDataset
from DeblurResNet.deblur_resnet import DeblurResNet

DATASET_DIR = '/home/user/gkazanta/GOPRO_Large'
if platform.system() == 'Windows':
    DATASET_DIR = 'C:\\Work\\DL\\datasets\\GOPRO_Large'


def train(restore_model=None, epoch=0):
    # Create data sets for training & for testing
    dataset_train = DeblurDataset(train=True, root_dir=DATASET_DIR, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    train_dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=1)

    dataset_test = DeblurDataset(train=False, root_dir=DATASET_DIR, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=1)

    # Select device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE WILL BE USED: ", device)

    # Create our model
    net = DeblurResNet()

    # Use data parallel for multi GPU training
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=[0, 1, 2])
    net.to(device)

    logger = Logger('./logs')

    # Restore model if given
    if restore_model is not None:
        net.load_state_dict(torch.load(restore_model, map_location={'cuda:0': 'cpu'}))
        print("Model {} was restored".format(restore_model))

    # Setup loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5*(1e-5))

    def rmse(y, y_hat):
        """Compute root mean squared error"""
        return torch.sqrt(torch.mean((y - y_hat).pow(2)))

    # Start training
    last_accuracy = None
#    epoch = 0
    while True:
        print("Running {} epoch".format(epoch))
        running_loss = 0.0
        start_time = time.time()
        start_trening_time = time.time()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs
            inputs, reference = data['image'].float(), data['reference'].float()
            inputs = inputs.squeeze(0)
            reference = reference.squeeze(0)
            inputs, reference = inputs.to(device), reference.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, reference)
            loss.backward()
            optimizer.step()

            # print statistics
            print('.', end='', flush=True)
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                elapsed_time = time.time() - start_time
                print('[%d, %5d] loss: %.3lf ' % (epoch + 1, i + 1, running_loss), end='')
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                running_loss = 0.0
                start_time = time.time()
        epoch += 1

        for dl, type in zip([test_dataloader], ['test']):
            loss = 0
            total = 0
            with torch.no_grad():
                for data in dl:
                    images, reference = data['image'].float(), data['reference'].float()
                    images = images.squeeze(0)
                    reference = reference.squeeze(0)
                    images, reference = images.to(device), reference.to(device)

                    outputs = net(images)
                    total += len(outputs)
                    loss += rmse(outputs, reference)
            print('\nLoss of the network on the ' + type + ' set with ' + str(total) + ' test images: %f' % loss)
            # Tensorboard logging
            logging(logger, net, {'loss': running_loss, 'test': loss}, epoch)

        print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_trening_time)))

        torch.save(net.state_dict(), './model-frozen-{}'.format(epoch))

    print('Finished Training')
