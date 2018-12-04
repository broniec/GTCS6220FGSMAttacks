import sys
import helper_functions as hf
import torch
# import targeted_attack_w_centroids as targeted_attack
import targeted_attack_w_svm as targeted_attack
# import targeted_attack_w_knn as targeted_attack
# import centroids
import svm
# import knn
# centroids.load()
svm.train()
# knn.train_knn(15)


def run_targeted_experiment(fname):
    u_out = open(fname + ".txt", "w")
    u_out.write("img,original_class,target_class,num_iterations\n")
    data = hf.retrieve_semeion_data()
    model = torch.load('SemeionCNN98+Noise')
    fgst = targeted_attack.FastGradientSignTargeted(model, 0.02)
    size = len(data[0])

    for i in range(size):
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            o_image, o_class = hf.load_image(i)
            t_class = j
            iteration = fgst.generate(o_image, o_class, t_class)
            u_out.write("{},{},{},{}\n".format(i, o_class, t_class, iteration))
            u_out.flush()
        if i % 100 == 0:
            print(i)
    u_out.close()


if __name__ == '__main__':
    size = -1
    if len(sys.argv) > 2:
        size = int(sys.argv[2])
    run_targeted_experiment('bf_kernel_svm')
