import sys
import platform

from DeblurResNet.eval import eval, load_and_dump_to_onnx
from DeblurResNet.train import train


MODEL_PATH='/home/gkazanta/PyTorchModels/Hotdog_classifier/DeblurResNet/model-frozen-3'
if platform.system() == 'Windows':
    MODEL_PATH = 'PATH TO MODEL'


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'eval':
        eval(MODEL_PATH, sys.argv[2])
    elif sys.argv[1] == 'save_to_onnx':
        load_and_dump_to_onnx(MODEL_PATH)
    else:
        print("Unsupported key {}".format(sys.argv[1]))
