import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models


def retreive_semeion_data(file="../../data/semeion.data", generate_images=False):
    orig_data = np.loadtxt(file)
    data = (orig_data[:, :256]).astype('uint8')
    # data = (orig_data[:, :256]).astype('uint8')
    labels = np.nonzero(orig_data[:, 256:])[1]
    # data = np.invert(data)
    data = np.reshape(data, (-1, 16, 16))

    # Used to determine global mean and std for preprocessing and recreation functions

    # mean = [np.mean(m) for m in data[:]/255]
    # mean = np.mean(mean)
    # std = [np.std(m) for m in data[:]/255]
    # std = np.std(std)
    # print(mean)
    # print(std)

    if generate_images:
        id = 0
        for img in data:
            cv2.imwrite("../data/images/" + str(id) + "_" + str(labels[id]) + ".jpg", img)
            print(id)
            id += 1

    return data, labels


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 16 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """

    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (16, 16))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H

    im_as_arr = im_as_arr[0] # Only need one layer since this is a binary image
    im_as_arr /= 255

    # Normalize
    reverse_mean = 0.671
    reverse_std = .022
    im_as_arr -= reverse_mean
    im_as_arr /= reverse_std

    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()

    # Add two more channel to the beginning. Tensor shape = 1,1,16,16
    im_as_ten.unsqueeze_(0)
    im_as_ten.unsqueeze_(1)
    # print(im_as_ten.shape)

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

    reverse_mean = -0.671
    reverse_std = 1/.022
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    recreated_im = np.concatenate((recreated_im, recreated_im, recreated_im), 0)
    for c in range(3):
        recreated_im[c] /= reverse_std
        recreated_im[c] -= reverse_mean
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im
