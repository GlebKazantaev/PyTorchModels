import sys

from DeblurResNet.eval import eval, load_and_dump_to_onnx
from DeblurResNet.train import train

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'eval':
        eval("C:\\Work\\PyTorchModels\\Hotdog_classifier\\model-frozen-78", 'C:\\Users\\gkazanta\\Pictures\\deblur_test.png')
    elif sys.argv[1] == 'save_to_onnx':
        load_and_dump_to_onnx("C:\Work\PyTorchModels\Hotdog_classifier\model-frozen-1")
    else:
        print("Unsupported key {}".format(sys.argv[1]))
