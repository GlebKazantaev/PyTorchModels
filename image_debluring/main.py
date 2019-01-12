import optparse

from deblur_image import DeblurImageEngine
from deblur_resnet import DeblurResnetModel

parser = optparse.OptionParser()

parser.add_option('-m', '--input_model',
                  action="store",
                  dest="input_model",
                  help="Path to saved PyTorch model to restore",
                  default=None)

parser.add_option('-i', '--image',
                  action="store",
                  dest="input_image",
                  help="Path to image to be evaluated",
                  default="")

parser.add_option('-d', '--dataset_root',
                  action="store",
                  dest="dataset_root",
                  help="Path to data set (/home/user/gkazanta/GOPRO_Large or C:\\Work\\DL\\datasets\\GOPRO_Large",
                  default="")

parser.add_option('-f', '--image_tiling',
                  action="store_true",
                  dest="use_tiling",
                  help="Evaluate given model with tiled image",
                  default=False)

parser.add_option('-e', '--eval',
                  action="store_true",
                  dest="eval",
                  help="Evaluate given model with given image (By default image will be resized. Use -f to process tiled image)",
                  default=False)

parser.add_option('-t', '--train',
                  action="store_true",
                  dest="train",
                  help="Start training. If -m specified it will start training with given model",
                  default=False)

parser.add_option('-l', '--loss',
                  action="store",
                  dest="loss",
                  help="Supported losses: MSE",
                  default="MSE")

parser.add_option('-g', '--gpu_ids',
                  action="store",
                  dest="gpu_ids",
                  help="Use to specify GPU ids to use (ex. 0,1)",
                  default=None)


if __name__ == "__main__":
    options, args = parser.parse_args()

    h, w = 128, 128

    engine = DeblurImageEngine(DeblurResnetModel, h, w)

    if options.train:
        engine.train(dataset_dir=options.dataset_root,
                     loss_type=options.loss,
                     gpu_ids=options.gpu_ids)
    elif options.eval:
        engine.deblur_image(restore_model=options.input_model,
                            img_path=options.input_image,
                            resize=not options.use_tiling)
    else:
        print("Unsupported keys!")
