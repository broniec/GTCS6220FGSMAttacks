import numpy as np
from sklearn import svm


model = None


def _svm_train(data, labels):
    global model
    model = svm.SVC(kernel='rbf')
    model.fit(data, labels)
    y = model.predict(data)


def train():
    raw_data = np.loadtxt('cnn_out_weights', dtype=float)
    data = np.zeros((raw_data.size, raw_data[0].size - 1))
    labels = np.zeros(raw_data.size)
    data, labels = np.hsplit(raw_data, [raw_data[0].size - 1])
    labels = np.resize(labels, labels.size)
    _svm_train(data, labels)


def test(input):
    global model
    y = model.predict(input)
    return y


if __name__ == '__main__':
    train(10)
