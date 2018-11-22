import torch
import targeted_attack_w_svm as t_a


if __name__ == '__main__':
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print()
    print()
