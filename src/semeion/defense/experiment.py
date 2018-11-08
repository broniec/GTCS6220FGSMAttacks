import torch
import sys
import semeion as s
import targeted_attack_w_knn as targeted_attack
import helper_functions as hf
import knn


def run_targeted_experiment(size):
    u_out = open("targeted_experiment_out.txt", "w")
    u_out.write("img,original_class,target_class,predicted_class,num_iterations\n")
    knn.train_knn(5)
    data = hf.retrieve_semeion_data()
    model = hf.get_model()
    fgst = targeted_attack.FastGradientSignTargeted(model, 0.02)

    if size == -1:
        size = len(data[0])

    for i in range(size):
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            o_image, o_class = hf.load_image(i)
            t_class = j
            iteration = fgst.generate(o_image, o_class, t_class)
            u_out.write("{},{},{},{}\n".format(i, o_class, t_class, iteration))
    u_out.close()


if __name__ == '__main__':
    size = -1
    if len(sys.argv) > 2:
        size = int(sys.argv[2])
    run_targeted_experiment(size)
