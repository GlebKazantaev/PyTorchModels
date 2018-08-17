import torch
from skimage import io
from torchvision import transforms

from DeblurResNet.deblur_resnet import DeblurResNet
from DeblurResNet.dataset import random_crop_image
from common import save_to_onnx
from PIL import Image


def eval(restore_model, img_path='C:\\Work\\DL\\datasets\\GOPRO_Large\\test\\GOPR0384_11_00\\blur\\000001.png'):
    net = DeblurResNet()
    net.load_state_dict(torch.load(restore_model, map_location={'cuda:0': 'cpu'}))

    img = Image.open(img_path)#random_crop_image(Image.open(img_path), 64, 64)

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).float()
    img = img.unsqueeze(0)

    out = net(img)

    out = out.squeeze(0)
    out = transforms.functional.to_pil_image(out)
    out.save('out.bmp')

    img = img.squeeze(0)
    img = transforms.functional.to_pil_image(img)
    img.save('input.bmp')

    print("success")


def load_and_dump_to_onnx(restore_model):
    net = DeblurResNet()
    net.load_state_dict(torch.load(restore_model, map_location={'cuda:0': 'cpu'}))
    print("Model {} was restored".format(restore_model))
    save_to_onnx(net, "DeblurNet")
