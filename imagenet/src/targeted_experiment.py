import os
import cv2
import torch
import pretrainedmodels as p
import fast_gradient_sign_targeted as fgst


# noinspection PyBroadException
def main():
    output = open("resnet152targetedattack.txt", "w")
    output.write("iteration,original_class,target_class,predicted_class,confidence\n")
    model = torch.load('alexnetNoise')
    attack = fgst.FastGradientSignTargeted(model, 0.01)
    our_path = "tiny-imagenet-200/train/"

    for root, dirs, files in os.walk(our_path, topdown=True):
        for name in dirs:
            try:
                original_class = int(name)
            except:
                continue
            for filename in os.listdir(our_path + name + "/images/"):
                if filename.endswith(".JPEG"):
                    for t in dirs:
                        try:
                            target_class = int(t)
                        except:
                            continue
                        original_image = cv2.imread(our_path + name + "/images/" + filename, 1)
                        iteration, i, j, k, conf = attack.generate(original_image, original_class, target_class)
                        output.write("{},{},{},{},{}\n".format(iteration, i, j, k, conf))
                        output.flush()
                        break


if __name__ == '__main__':
    main()
