import matplotlib.pyplot as plt
import numpy as np


def graph(conf_0, conf_1):
    data = np.loadtxt('cnn_out_weights_for_graph', dtype=float)
    x, y, c = np.hsplit(data, [1, 2])
    x = np.append(x, [conf_0])
    y = np.append(y, [conf_1])
    c = np.append(c, -1)
    plt.scatter(x, y, c=c)
    plt.show()


if __name__ == '__main__':
    graph(-20, -20)
