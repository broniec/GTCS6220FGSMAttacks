import numpy as np


centroids = None


def test(input):
    if centroids is None:
        load()
    min_index = 0
    min_distance = np.linalg.norm(centroids[0] - input)
    for i in range(1, 10):
        distance = np.linalg.norm(centroids[i] - input)
        if distance < min_distance:
            min_index = i
            min_distance = distance
    return min_index


def load():
    global centroids
    centroids = np.loadtxt('semeion_centroids', dtype=float)


if __name__ == '__main__':
    x = np.zeros(10)
    print(test(x))
