import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import pretrainedmodels as p
import fast_gradient_sign_untargeted as fgsu
from misc_functions import preprocess_image, recreate_image, get_params
import cv2
#
# print(p.model_names)
# print(p.pretrained_settings['nasnetalarge'])

# model = p.squeezenet1_0(num_classes=1000, pretrained='imagenet')
# model.eval()
#
# torch.save(model, 'imagenet-squeezenet1_0')

def run_experiment():
    output = open("resnet152untargetedattack.txt", "w")
    output.write("original_class,predicted_class,num_iteration,confidence\n")
    pretrained_model = p.resnet152(num_classes=1000, pretrained='imagenet')
    FGS_untargeted = fgsu.FastGradientSignUntargeted(pretrained_model, 0.01)
    our_path = "tiny-imagenet-200/train/"



    for root, dirs, files in os.walk(our_path, topdown=True):
        # for name in files:
        #     print(os.path.join(root, name))
        for name in dirs:
            count = 0
            try:
                target_class = int(name)
                # print(name + "/images"
            except:
                continue
            for filename in os.listdir(our_path + name + "/images/"):
                if filename.endswith(".JPEG"):
                    if count > 99:
                        break
                    count += 1
                    # original_image = cv2.imread("tiny-imagenet-200/train/" + name + "/images/n01443537_{}.JPEG".format(i), 1)
                    original_image = cv2.imread(our_path + name + "/images/" + filename, 1)

                    # # print("tiny-imagenet-200/train/n01443537/images/n01443537_{}.JPEG".format(i))
                    # ppimage = preprocess_image(original_image)
                    # target_class_num = int(target_class[1:])
                    pred, iter, conf = FGS_untargeted.generate(original_image, target_class)
                    output.write("{},{},{},{}\n".format(target_class, pred, iter, conf))
            # print(os.path.join(root, name))


        # for file in files:
        #     print(len(path) * '---', file)


    #
    # for i in range(10):
    #     original_image = cv2.imread("tiny-imagenet-200/train/n01443537/images/n01443537_{}.JPEG".format(i), 1)
    #     # print("tiny-imagenet-200/train/n01443537/images/n01443537_{}.JPEG".format(i))
    #     ppimage = preprocess_image(original_image)
    #     target_class = "n01443537"
    #     target_class_num = int(target_class[1:])
    #
    #
    #
    #     FGS_untargeted.generate(original_image, 1)





if __name__ == '__main__':
        run_experiment()
