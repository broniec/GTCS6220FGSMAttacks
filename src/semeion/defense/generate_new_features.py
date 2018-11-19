import helper_functions as hf
import numpy as np


class RunImageOnModel:
    """
        Fast gradient sign untargeted adversarial attack, maximizes the target class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        self.alpha = alpha

    def run(self, image):
        processed_image = hf.preprocess_image(image)
        processed_image.grad = None
        out = self.model(processed_image)
        return out.data.numpy()[0]


def generate_new_features():
    net = RunImageOnModel(hf.get_model(), 0.02)
    hf.load_data()
    length = len(hf.data[0])
    x = np.zeros((length, 11))
    for i in range(length):
        image, label = hf.load_image(i)
        features = net.run(image)
        x[i][0:10] = np.copy(features)
        x[i][10] = label
    np.savetxt('cnn_out_weights', x)


if __name__ == '__main__':
    generate_new_features()
