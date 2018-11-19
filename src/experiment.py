import semeion as s
import torch
import sys
import fast_gradient_sign_untargeted as fgsu
import misc_functions_2 as mf
from misc_functions import preprocess_image, recreate_image, get_params
import numpy as np

# batch_sizes = [1, 2, 4, 8, 16, 32, 64]
learning_rates = [0.02, 0.03, 0.05, 0.07, 0.09]
linear_layer_sizes = [50, 55, 60, 65, 70, 75, 80]


def run_experiments():
    for bs in [64]:
        for lr in learning_rates:
            for lls in linear_layer_sizes:
                print(bs, lr, lls)
                s.run(bs, 1000, lr, lls)


def run_experiment():
    model = s.run(64, 327, 0.0425, 75)
    torch.save(model, 'SemeionCNN98+Noise')

def run_untargeted_experiment():
    u_out = open("untargeted_eperiment_out.txt", "w")
    u_out.write("img,original_class,predicted_class,num_iterations,confidence\n")
    data = mf.retreive_semeion_data()

    num_data_points = len(data[0])

    for i in range(num_data_points):
        (original_image, prep_img, target_class, _, pretrained_model) =\
            get_params(i)

        FGS_untargeted = fgsu.FastGradientSignUntargeted(pretrained_model, 0.02)
        pred,iter,conf = FGS_untargeted.generate(original_image, target_class)
        u_out.write("{},{},{},{},{}\n".format(i,target_class,pred,iter,conf))
    u_out.close()

if __name__ == '__main__':

    if len(sys.argv) == 2:
        run_untargeted_experiment()
    else:
        run_experiment()
