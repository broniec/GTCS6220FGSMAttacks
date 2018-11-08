import numpy as np
import cv2
import torch
from torch import nn
from torch.autograd import Variable
import helper_functions as hf
import knn


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
        # Start iteration
        for i in range(100):
            # print('Iteration:', str(i))
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
            confirmation_prediction = int(knn.test_knn(confirmation_out.data)[0])
            if confirmation_prediction == target_class or i == 99:
                # print('Original image was predicted as:', org_class,
                #       'with adversarial noise converted to:', confirmation_prediction)
                return i


if __name__ == '__main__':
    model = hf.get_model()
    o_image, o_class = hf.load_image(0)
    t_class = (o_class + 2) % 10

    knn.train_knn(3)
    fgst = FastGradientSignTargeted(model, 0.02)
    fgst.generate(o_image, o_class, t_class)
