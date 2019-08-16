import torch
from torch.autograd import Variable
from models.simple_net import SimpleNet
import onnx
from onnx_tf.backend import prepare
import os


def convert_to_tensorflow(pytorch_model_path, model_class):
    """
    Converts a trained pytorch model to a tensorflow model using ONNX
    :param pytorch_model_path: Path to pytorch model
    """

    # Setup filenames
    fileroot = os.path.basename(pytorch_model_path).split('.')[0]
    dirname = os.path.dirname(pytorch_model_path)
    onnx_path = '{}.onnx'.format(os.path.join(dirname, fileroot))
    tf_path = '{}.pb'.format(os.path.join(dirname, fileroot))

    print("Saving onnx file to {}".format(onnx_path))
    print("Saving onnx file to {}".format(tf_path))
    
    print("Loading trained model...")
    trained_model = model_class()
    trained_model.load_state_dict(torch.load('output/simple_net.pth'))

    # Export the trained model to ONNX
    print("Exporting model to ONNX...")
    dummy_input = Variable(torch.randn(1, 3, 64, 64))
    torch.onnx.export(trained_model, dummy_input, onnx_path)

    # Create Tensorflow Representation
    print("Creating tensorflow representation...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('output/mnist.pb')


if __name__ == '__main__':

    pytorch_model_path = '/home/robbie/Desktop/virtual_box_files/personal/RotorTrainer/output/simple_net.pth'
    model_class = SimpleNet

    convert_to_tensorflow(pytorch_model_path, model_class)