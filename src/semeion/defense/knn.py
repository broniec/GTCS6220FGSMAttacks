import numpy as np
from sklearn import neighbors


model = None


def _knn_train(num_neighbors, data, labels):
    global model
    model = neighbors.KNeighborsClassifier(num_neighbors)
    model.fit(data, labels)


def train_knn(num_neighbors):
    raw_data = np.loadtxt('cnn_out_weights', dtype=float)
    print(raw_data.size)
    data = np.zeros((raw_data.size, raw_data[0].size - 1))
    labels = np.zeros(raw_data.size)
    data, labels = np.hsplit(raw_data, [raw_data[0].size - 1])
    labels = np.resize(labels, labels.size)
    _knn_train(num_neighbors, data, labels)


def test_knn(instance):
    global model
    y = model.predict(instance)
    return y


if __name__ == '__main__':
    train_svm(10)
