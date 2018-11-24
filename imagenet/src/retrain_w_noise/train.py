import os
import time
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pretrainedmodels as p
from misc_functions import preprocess_image, recreate_image, get_params


class GenerateNoisyModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def train(self, original_image, label, optimizer):
        processed_image = preprocess_image(original_image)
        processed_image.grad = None
        out = self.model(processed_image)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()

    def save(name):
        torch.save(self.model, name + '+Noise')


def go_net():
    counter = 0
    model = p.alexnet(num_classes=1000, pretrained='imagenet')
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    gw = GenerateNoisyModel(model)
    our_path = "../tiny-imagenet-200/train/"

    for root, dirs, files in os.walk(our_path, topdown=True):
        for name in dirs:
            for filename in os.listdir(our_path + name + "/images/"):
                if filename.endswith(".JPEG"):
                    original_image = cv2.imread(our_path + name + "/images/" + filename, 1).astype(float)
                    original_image += np.random.normal(0.0, 0.1, (64, 64, 3))
                    out = gw.train(original_image, torch.from_numpy(np.full(1, int(name))), optimizer)
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                        print(time.ctime())
        break
    gw.save('alexnet')


if __name__ == '__main__':
    go_net()
