import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from common import Logger, logging
from dataset import DeblurDataset


class DeblurImageEngine:
    loss_map = {
        'MSE': nn.MSELoss(),
    }

    def __init__(self, engine, h, w):
        self.engine = engine
        self.h = h
        self.w = w

    def __eval(self, net, img):
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).float()
        img = transforms.functional.normalize(img, mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        img = img.unsqueeze(0)

        out = net(img)

        out = out.squeeze(0)

        out_v = Variable(out, requires_grad=False).cpu()
        out_img = transforms.functional.to_pil_image(out_v)
        return out_img

    def deblur_image(self, restore_model, img_path, resize=False, sample_height=None, sample_width=None):
        if sample_height is None:
            sample_height = self.h

        if sample_width is None:
            sample_width = self.w

        net = self.engine()

        epoch = restore_model.split('-')[-1]
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(restore_model, map_location='cpu'))
        print("Model {} was restored".format(restore_model))

        net.eval()

        img = Image.open(img_path)
        if resize:
            try:
                img = img.resize((sample_height, sample_width), Image.ANTIALIAS)
                print("Image was resized to {}".format((sample_height, sample_width)))
            except:
                raise RuntimeError("Cant resize image {} to {}".format(img_path, (sample_height, sample_width)))

        res_image = []
        step_h = sample_height
        step_w = sample_width
        w = 0
        h = 0
        flag = False
        while h + step_h <= img.height:
            while w + step_w <= img.width:
                crop_image = img.crop((w, h, w + step_w, h + step_h))
                out_img = self.__eval(net, crop_image)
                np_out = np.array(out_img)

                if len(res_image) == 0:
                    res_image.append(np_out)
                else:
                    if flag:
                        res_image.append(np_out)
                    else:
                        res_image[-1] = np.concatenate((res_image[-1], np_out), axis=1)
                    flag = False
                w += step_w
            h += step_h
            w = 0
            flag = True

        out_img = None
        for t in res_image:
            if out_img is None:
                out_img = t
            else:
                out_img = np.concatenate((out_img, t), axis=0)

        out_img_path = os.path.join(os.path.dirname(img_path),
                                    os.path.basename(img_path).split('.')[:-1][0] + "_out_{}.bmp".format(epoch))

        out_img = Image.fromarray(out_img)
        out_img.save(out_img_path)

        print(out_img_path)
        print("success")

    def train(self, dataset_dir, loss_type, gpu_ids=None, restore_model=None, epoch=1, pref="model"):
        # Set loss function
        if loss_type not in self.loss_map:
            raise RuntimeError("Unregistred loss {}".format(loss_type))
        criterion = self.loss_map[loss_type]

        # Check gpu ids
        if gpu_ids is not None:
            try:
                gpu_ids = [int(x) for x in gpu_ids.split(',')]
            except:
                raise RuntimeError("Wrong gpu_ids attribute {}".format(gpu_ids))

        # Create data sets for training & for testing
        dataset_train = DeblurDataset(train=True, root_dir=dataset_dir, transform=transforms.Compose([
            transforms.ToTensor()
        ]), h=self.h, w=self.w)
        train_dataloader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=1)

        dataset_test = DeblurDataset(train=False, root_dir=dataset_dir, transform=transforms.Compose([
            transforms.ToTensor()
        ]), h=self.h, w=self.w)
        test_dataloader = DataLoader(dataset_test, batch_size=2, shuffle=True, num_workers=1)

        # Select device for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE WILL BE USED: ", device)

        # Create our model
        net = self.engine()

        # Use data parallel for multi GPU training
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            if gpu_ids is not None:
                net = nn.DataParallel(net, device_ids=gpu_ids)
            else:
                net = nn.DataParallel(net)

        net.to(device)

        logger = Logger('./logs_{}'.format(pref))

        # Restore model if given
        if restore_model is not None:
            net.load_state_dict(torch.load(restore_model, map_location={'cuda:0': 'cpu'}))
            print("Model {} was restored".format(restore_model))
            try:
                epoch = int(restore_model.split('-')[-1])
            except:
                print("Cant extract epoch id from {} file! Will be used 0".format(restore_model))

        # Setup loss function
        # criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=5 * (1e-6))

        # Start training
        last_accuracy = None
        while True:
            print("Running {} epoch".format(epoch))
            running_loss = 0.0
            epoch_loss = 0.0
            start_time = time.time()
            start_training_time = time.time()
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs
                inputs, reference = data['image'].float(), data['reference'].float()
                inputs = inputs.squeeze(1)
                reference = reference.squeeze(1)
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
                epoch_loss += loss.item()
                if i % 10 == 9:  # print every 100 mini-batches
                    elapsed_time = time.time() - start_time
                    print('[%d, %5d] loss: %.3lf ' % (epoch, i + 1, running_loss), end='')
                    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    running_loss = 0.0
                    start_time = time.time()
            if epoch % 1 == 0:
                for dl, type in zip([test_dataloader], ['test']):
                    loss = 0
                    total = 0
                    reference, outputs = None, None
                    with torch.no_grad():
                        for data in dl:
                            images, reference = data['image'].float(), data['reference'].float()
                            images = images.squeeze(1)
                            reference = reference.squeeze(1)
                            images, reference = images.to(device), reference.to(device)

                            outputs = net(images)
                            total += len(outputs)
                            loss += criterion(outputs, reference)
                    print('Loss of the network on the test dataset: %.3lf' % loss)
                    # print('\nLoss of the network on the ' + type + ' set with ' + str(total) + ' test images: %f' % loss)
                    # TensorBoard logging
                    logging(logger,
                            net,
                            {'loss': epoch_loss, 'test': loss, 'ref_img': reference, 'outputs': outputs},
                            epoch)
                print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_training_time)))
            torch.save(net.state_dict(), './{}-frozen-{}x{}-{}-{}'.format(pref, self.h, self.w, loss_type, epoch))
            epoch += 1

        print('Finished Training')
