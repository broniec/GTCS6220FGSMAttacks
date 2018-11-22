import os
import cv2
import numpy as np
import pretrainedmodels as p
from misc_functions import preprocess_image, recreate_image, get_params
import svm


class GenerateWeights:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, original_image):
        processed_image = preprocess_image(original_image)
        processed_image.grad = None
        out = self.model(processed_image)
        return out.data.numpy()[0]


def go_net():
    counter = 0
    weights = np.zeros((20000, 1000), dtype=float)
    model = p.resnet152(num_classes=1000, pretrained='imagenet')
    gw = GenerateWeights(model)
    our_path = "tiny-imagenet-200/train/"

    for root, dirs, files in os.walk(our_path, topdown=True):
        print(dirs)
        for name in dirs:
            for filename in os.listdir(our_path + name + "/images/"):
                if filename.endswith(".JPEG"):
                    original_image = cv2.imread(our_path + name + "/images/" + filename, 1)
                    out = gw.generate(original_image)
                    weights[counter] = out
                    counter += 1
                    if counter % 100 == 0:
                        break
        break
    np.savetxt("res_152_w(100).np", weights)


if __name__ == '__main__':
    go_net()
