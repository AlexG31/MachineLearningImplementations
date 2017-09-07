#encoding:utf8
import os, sys, pdb
import codecs, matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier


def Normalize(y):
    max_y = np.max(y)
    min_y = np.min(y)
    if max_y - min_y < 1e-6:
        return y
    return (y - min_y) / (max_y - min_y)

def gen_classification_samples():
    x = np.ones((100, 1))
    x = np.append(x, np.zeros((100, 1)), axis = 0)
    x = np.append(x, 1 - x, axis = 1)
    y = np.ones((200, 1))
    x = np.append(x, np.zeros((100, 2)), axis = 0)
    x = np.append(x, np.ones((100, 2)), axis = 0)
    y = np.append(y, np.zeros((200, 1)), axis = 0)
    return (x, y)

def plot_classification_result(x, y, title = None, show = False,
        figure_id = None,
        color_pair = None):
    x_A = x[np.array(y) == 'A']
    x_B = x[np.array(y) == 'B']

    if color_pair is None:
        color_pair = ('y', 'r')
    if figure_id is None:
        plt.figure()
    else:
        plt.figure(figure_id)
    color_mat = ['r' if val == 0 else 'b' for val in y]
    plt.plot(x_A[:, 0], x_A[:, 1], 'ro', color = color_pair[0],
            alpha = 0.3, markersize = 12)
    plt.plot(x_B[:, 0], x_B[:, 1], 'ro', color = color_pair[1],
            alpha = 0.3, markersize = 12)
    if title is not None:
        plt.title(title)
    plt.grid(True)
    if show:
        plt.show()

def test_classification():
    x, y = gen_classification_samples()
    x += (np.random.rand(x.shape[0], x.shape[1]) - 0.5) * 0.5
    y = np.squeeze(y)
    y = ['A' if abs(val) < 1e-6 else 'B' for val in y]

    # NN classification
    layers = np.ones(1) * 100
    print layers.shape
    layers = tuple([int(val) for val in layers.tolist()])
    nn = MLPClassifier(hidden_layer_sizes = layers,
            learning_rate = 'constant',
            # solver = 'lbfgs',
            warm_start = False,
            alpha = 0.75,
            max_iter = 300)
    nn.fit(x, y)
    x1 = np.random.rand(1000, 2) * 2 - 0.5
    y1 = nn.predict(x1)
    plot_classification_result(x1, y1, title = 'Prediction', figure_id = 1, color_pair = ('g', 'm'))
    plot_classification_result(x, y, title = 'Trained NN', show = True, figure_id = 1)
    
def test_sklearn():
    '''Run example NN from sklearn.'''
    x = np.linspace(-10, 10, 100)
    y = -2.0 * x * x * x + 1.2 * x*x - 2.0 * x + 23
    y += np.random.rand(100) * 100
    x = Normalize(x)
    y = Normalize(y)

    # Train NN regressor
    layers = np.ones(5) * 100
    print layers.shape
    layers = tuple([int(val) for val in layers.tolist()])

    nn = MLPRegressor(hidden_layer_sizes = layers,
            learning_rate = 'constant',
            solver = 'lbfgs',
            warm_start = False,
            # alpha = 0.15,
            max_iter = 300)
    x = x.reshape(-1, 1)
    x2 = x * x
    x_expand = np.append(x, x2, axis = 1)
    x3 = x * x * x
    x_expand = np.append(x_expand, x3, axis = 1)
    x4 = x * x * x * x
    x_expand = np.append(x_expand, x4, axis = 1)
    # const_array = np.ones((100, 1))
    # x_expand = np.append(x_expand, const_array, axis = 1)
    nn.fit(x_expand, y)
    y1 = nn.predict(x_expand)
    
    plt.figure()
    plt.plot(x, y, alpha = 0.7, mfc = 'none', mec = 'r',
            markersize = 12, marker = 'o', linestyle = "none")
    x = np.squeeze(x)
    print x.shape
    plt.plot(x, y1, alpha = 0.8, color = 'b', lw = 1)
    plt.grid(True)
    plt.show()


    
if __name__ == '__main__':
    # test_sklearn()
    test_classification()
