import numpy as np
import cv2
import torch
from torch import nn
from torch.autograd import Variable
import helper_functions as hf


class FastGradientSignTargeted():
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
        # Start iteration
        for i in range(100):
            print('Iteration:', str(i))
            processed_image.grad = None
            out = self.model(processed_image)
            pred_loss = ce_loss(out, im_label_as_var)
            pred_loss.backward()

            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            processed_image.data = processed_image.data - adv_noise

            recreated_image = hf.recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = hf.preprocess_image(recreated_image)
            confirmation_out = self.model(prep_confirmation_image)
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            confirmation_prediction = confirmation_prediction.numpy()[0]
            if confirmation_prediction == target_class or i == 99:
                print('Original image was predicted as:', org_class,
                      'with adversarial noise converted to:', confirmation_prediction,
                      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                return confirmation_prediction, i, confirmation_confidence

        # noise_image = original_image - recreated_image
        # original_image = np.resize(original_image, (16, 16, 1))
        # original_image *= 255
        # original_image = np.dstack((original_image, original_image, original_image))
        # noise_image = np.resize(noise_image, (16, 16, 1))
        # noise_image *= 255
        # noise_image = np.dstack((noise_image, noise_image, noise_image))
        # recreated_image = np.resize(recreated_image, (16, 16, 1))
        # recreated_image *= 255
        # recreated_image = np.dstack((recreated_image, recreated_image, recreated_image))

        # # orig. image
        # cv2.imwrite('../generated/targeted/i_' + str(org_class) + '_' +
        #             str(confirmation_prediction) + '.jpg', original_image)
        # # final image
        # cv2.imwrite('../generated/targeted/f_' + str(org_class) + '_' +
        #             str(confirmation_prediction) + '.jpg', noise_image)
        # # noise image
        # cv2.imwrite('../generated/targeted/n_' + str(org_class) + '_' +
        #             str(confirmation_prediction) + '.jpg', recreated_image)


if __name__ == '__main__':
    model = hf.get_model()
    o_image, o_class = hf.load_image(0)
    t_class = (o_class + 2) % 10

    fgst = FastGradientSignTargeted(model, 0.02)
    fgst.generate(o_image, o_class, t_class)
