import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import torch
from torch.autograd import Variable

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def logging(logger, net, info, step):
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)
    #info = {'loss': loss.item(), 'accuracy': accuracy.item()}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step + 1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

    # # 3. Log training images (image summary)
    # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}
    #
    # for tag, images in info.items():
    #     logger.image_summary(tag, images, step + 1)


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


def split_images(path_with_images, h, w):
    images_list = os.listdir(path_with_images)
    for image_path in images_list:
        img = cv2.imread(image_path)
        h, w = img.shape
