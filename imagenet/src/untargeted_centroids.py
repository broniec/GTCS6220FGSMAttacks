import pretrainedmodels as p
import numpy as np
import torch
import os
import cv2
from torch import nn
from torch.autograd import Variable
from misc_functions import preprocess_image, recreate_image, get_params
import centroids


class UntargetedwCentroids():
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        self.alpha = alpha

    def generate(self, original_image, im_label):
        im_label_as_var = Variable(torch.from_numpy(np.asarray([im_label]))).long()
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(50):
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # print("out:   ", out)
            # print("im_label_as_var     ", im_label_as_var)
            # Calculate CE loss
            pred_loss = ce_loss(out, im_label_as_var)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add Noise to processed image
            processed_image.data = processed_image.data + adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            confirmation_out = self.model(prep_confirmation_image)
            confirmation_prediction = int(centroids.test(confirmation_out.data))
            if confirmation_prediction != im_label:
                return i, confirmation_prediction
        return 49, confirmation_prediction


if __name__ == '__main__':
    attack = UntargetedwCentroids(p.alexnet(), 0.01)
    print("alexnet")
    output = open("resnet161_untargeted_w_svm(100).txt", "w")
    output.write("iteration,original_class,predicted_class\n")
    our_path = "tiny-imagenet-200/train/"
    counter = 0
    for root, dirs, files in os.walk(our_path, topdown=True):
        for name in dirs:
            for filename in os.listdir(our_path + name + "/images/"):
                if filename.endswith(".JPEG"):
                    original_image = cv2.imread(our_path + name + "/images/" + filename, 1)
                    iteration, prediction = attack.generate(original_image, int(name))
                    output.write("{},{},{}\n".format(iteration, name, prediction))
                    output.flush()
                    counter += 1
                    if counter % 100 == 0:
                        print(counter)
                        break
        break
