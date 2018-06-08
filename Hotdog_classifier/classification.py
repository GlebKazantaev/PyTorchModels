import optparse
from vgg_net import vgg_train, vgg_eval_model
from simple_net import simple_net


parser = optparse.OptionParser()

parser.add_option('-r', '--restore_model',
    action="store", dest="restore_model",
    help="Path to PyTorch model to restore", default="")

parser.add_option('-d', '--dump',
    action="store", dest="dump",
    help="Dump PyTorch model to ONNX", default="")

parser.add_option('-p', '--dataset_path',
    action="store", dest="dataset_path",
    help="Path to dataset root dif", default="")

parser.add_option("-e", action="store_true", default=False, dest="eval", help="Evaluate VGG model (works only with -r)")



if __name__ == "__main__":
    options, args = parser.parse_args()

    if len(options.dataset_path) == 0:
        print("Please use -p <path_to_dataset>")
        exit(1)

    if options.eval and len(options.restore_model) == 0:
        print("Please use -r <path_to_trained_model>")
        exit(1)

    if options.eval:
        vgg_eval_model(options.dataset_path, options.restore_model)
    else:
        vgg_train(options.dataset_path, options.restore_model, options.dump)
