import torch
import torch.nn as nn

from torchvision import transforms

from image_debluring.deblur_resnet import DeblurResNet
from common import save_to_onnx
from PIL import Image
from common import Logger, logging


def eval(restore_model, img_path):
    # Select device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE WILL BE USED: ", device)
    net = DeblurResNet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        net.to(device)
    if restore_model is not None:
        net.load_state_dict(torch.load(restore_model))#, map_location={'cuda:0': 'cpu'}))
        print("Model {} was restored".format(restore_model))

    net.eval()

    img = Image.open(img_path)#random_crop_image(Image.open(img_path), 64, 64)

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img_norm = transforms.functional.normalize(img ,mean=[0.5, 0.5, 0.5], std=[1, 1, 1]).float()
   # iimg = transform(img).float()
    img_norm = img_norm.unsqueeze(0)
    img = img.unsqueeze(0)

    out = net(img_norm)
    logger = Logger('/home/gkazanta/PyTorchModels/hotdog_classifier/image_debluring/logger')
    logging(logger, net, {'ref_img': img, 'outputs': out}, 1)

    out = out.squeeze(0)
    #out = transforms.functional.to_pil_image(out)
    from torch.autograd import Variable
    out_v=Variable(out, requires_grad=False).cpu()
    out_img=transforms.functional.to_pil_image(out_v)
    out_img.save('out.bmp')

    img = img.squeeze(0)
    img = transforms.functional.to_pil_image(img)
    img.save('input.bmp')

    print("success")


def load_and_dump_to_onnx(restore_model):
    net = DeblurResNet()
    net.load_state_dict(torch.load(restore_model, map_location={'cuda:0': 'cpu'}))
    print("Model {} was restored".format(restore_model))
    save_to_onnx(net, "DeblurNet")
