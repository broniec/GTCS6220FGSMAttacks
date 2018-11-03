import semeion as s
import torch

# batch_sizes = [1, 2, 4, 8, 16, 32, 64]
learning_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
linear_layer_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]


# def run_experiment():
#     for bs in [64]:
#         for lr in learning_rates:
#             for lls in linear_layer_sizes:
#                 print(bs, lr, lls)
#                 s.run(bs, 1000, lr, lls)


def run_experiment():
    model = s.run(64, 333, 0.05, 65)
    torch.save(model, 'SemeionCNN')


if __name__ == '__main__':
    run_experiment()
