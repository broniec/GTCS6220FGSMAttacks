"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import copy
import cv2
import numpy as np
import misc_functions_2 as mf

import torch
from torch.autograd import Variable
from torchvision import models


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # # Resize image
    # if resize_im:
    #     cv2im = cv2.resize(cv2im, (224, 224))
    # im_as_arr = np.float32(cv2im)
    # im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    # print (im_as_arr.shape)
    # im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # print(im_as_arr.shape)
    # # Normalize the channels
    # for channel, _ in enumerate(im_as_arr):
    #     im_as_arr[channel] /= 255
    #     im_as_arr[channel] -= mean[channel]
    #     im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(cv2im).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_ten = im_as_ten.view(-1, 1, 16, 16)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """

    return im_as_var.detach().numpy()

    # reverse_mean = [-0.485, -0.456, -0.406]
    # reverse_std = [1/0.229, 1/0.224, 1/0.225]
    # recreated_im = copy.copy(im_as_var.data.numpy()[0])
    # for c in range(3):
    #     recreated_im[c] /= reverse_std[c]
    #     recreated_im[c] -= reverse_mean[c]
    # recreated_im[recreated_im > 1] = 1
    # recreated_im[recreated_im < 0] = 0
    # recreated_im = np.round(recreated_im * 255)
    #
    # recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # # Convert RBG to GBR
    # recreated_im = recreated_im[..., ::-1]
    # return recreated_im


def get_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """

    data = mf.retreive_semeion_data()
    original_image = data[0][example_index]
    prep_img = data[0][example_index]
    target_class = data[1][example_index]
    file_name_to_export = None

    # x = recreate_image(im_as_ten = torch.from_numpy(original_image).float())
    # print(x)

    # Define model
    # pretrained_model = models.alexnet(pretrained=True)
    pretrained_model = torch.load('SemeionCNN98')
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)
