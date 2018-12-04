import numpy as np
import cv2
import torch
from torch import nn
from torch.autograd import Variable
import helper_functions as hf
import svm
import graph


class FastGradientSignTargeted:
    """
        Fast gradient sign untargeted adversarial attack, maximizes the target class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        self.alpha = alpha

    def generate(self, original_image, org_class, target_class):
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])).long())
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = hf.preprocess_image(original_image)
        for i in range(50):
            # print('Iteration:', str(i))
            processed_image.grad = None
            out = self.model(processed_image)
            pred_loss = ce_loss(out, im_label_as_var)
            pred_loss.backward()

            # graph.graph(out.data[0][0], out.data[0][1])

            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            processed_image.data = processed_image.data - adv_noise

            recreated_image = hf.recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = hf.preprocess_image(recreated_image)
            confirmation_out = self.model(prep_confirmation_image)
            confirmation_prediction = int(svm.test(confirmation_out.data)[0])
            # print(confirmation_out.data)
            # graph.graph(confirmation_out.data[0][0], confirmation_out.data[0][1])
            if confirmation_prediction == target_class or i == 2:
                noise_image = original_image - recreated_image
                noise_image = np.resize(noise_image, (16, 16, 1))
                noise_image = noise_image * 255
                noise_image = np.dstack((noise_image, noise_image, noise_image))
                recreated_image = np.resize(recreated_image, (16, 16, 1))
                recreated_image = recreated_image * 255
                recreated_image = np.dstack((recreated_image, recreated_image, recreated_image))
                cv2.imwrite('noise1.bmp', noise_image)
                # Write image
                cv2.imwrite('recreated1.bmp', recreated_image)
                return i


if __name__ == '__main__':
    model = torch.load('SemeionCNN98')
    o_image, o_class = hf.load_image(0)
    t_class = o_class + 1

    svm.train()
    fgst = FastGradientSignTargeted(model, 0.02)
    fgst.generate(o_image, o_class, t_class)
