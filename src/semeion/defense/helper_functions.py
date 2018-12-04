import cv2
import numpy as np
import torch
import torch.autograd as ag
from scipy import ndimage


data = None


def retrieve_semeion_data(file="../../../data/semeion.data", generate_images=False):
    orig_data = np.loadtxt(file)
    data = (orig_data[:, :256]).astype('uint8')
    labels = np.nonzero(orig_data[:, 256:])[1]
    data = np.reshape(data, (-1, 16, 16))

    if generate_images:
        id = 0
        for img in data:
            cv2.imwrite("../data/images/" + str(id) + "_" + str(labels[id]) + ".jpg", img)
            print(id)
            id += 1

    return data, labels


def get_model():
    return torch.load('SemeionCNN98+Noise')


def load_data():
    global data
    data = retrieve_semeion_data()


def load_image(index):
    global data
    if data is None:
        data = retrieve_semeion_data()
    return data[0][index], data[1][index]


def rem_noise(image):
    return ndimage.gaussian_filter(image, 0.3)


def preprocess_image(image):
    # image = rem_noise(image)
    im_as_ten = torch.from_numpy(image).float()
    im_as_ten.unsqueeze_(0)
    im_as_ten = im_as_ten.view(-1, 1, 16, 16)
    im_as_var = ag.Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(image):
    return image.clone().detach().numpy()
    # return image
