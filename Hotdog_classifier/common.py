import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.autograd import Variable


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.pause(1.1)


def save_to_onnx(net, model_name):
    input_shape = [1, 3, 64, 64]
    model_name += '.onnx'
    dummy_input = Variable(torch.randn(input_shape), requires_grad=True)
    torch.onnx.export(net, dummy_input, "{}".format(model_name), export_params=True)  # , verbose=True)
    print('[ INFO ] Success\n')
