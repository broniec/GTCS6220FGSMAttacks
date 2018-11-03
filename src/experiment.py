import semeion as s
import torch

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
    torch.save(model, 'SemeionCNN98')


if __name__ == '__main__':
    run_experiment()
