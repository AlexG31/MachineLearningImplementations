#encoding:utf8
import os, sys, pdb
import numpy as np
import NeuralCells as neuralcells
import matplotlib.pyplot as plt
import networkx as nx

def getNeuralCell(cell_type):
    if cell_type in ['identity', 'base']:
        return neuralcells.CellBase()
    elif cell_type == 'relu':
        return neuralcells.ReluCell()
    elif cell_type == 'sigmod':
        return neuralcells.SigmodCell()
    else:
        raise Exception('Unknown cell type: {}'.format(cell_type))

class SimpleNeuralNetwork(object):
    def __init__(self, layers_config = ((100, 'relu'),), alpha = 0.001):
        self.layers = self.getLayers(layers_config) 
        self.weight_layers = []
    
    def getLayers(self, layers_config):
        layers = []
        for neural_number, neural_type in layers_config:
            cell_layer = []
            for ind in xrange(0, neural_number):
                cell_layer.append(getNeuralCell(neural_type))
            layers.append(cell_layer)
        return layers
            
    def forward_propergate(self, x_in):
        if x_in.shape[1] != self.weight_layers[0].shape[0]:
            raise Exception('Input must have the same shape: x_in: {}'.format(x_in.shape))
        # Inputs at each layer
        layer_input_values = [x_in, ]
        current_layer_value = np.array(x_in)
        for ind in xrange(0, len(self.weight_layers)):
            current_layer_value = current_layer_value.dot(self.weight_layers[ind])

            if ind < len(self.layers):
                # Skip output layer
                cell = self.layers[ind][0]
                current_layer_value = cell.safe_f(current_layer_value)
                # Inputs before weights
                layer_input_values.append(current_layer_value)

        return current_layer_value, layer_input_values

    def back_propergate(self, x, y, learning_rate = 0.02):
        def loss_derivative(y, y_estimate):
            return y_estimate - y
        current_layer_value, layer_input_values = self.forward_propergate(x)
        error_layer = loss_derivative(y, current_layer_value)
        updated_weights = []
        for weight_layer in self.weight_layers:
            updated_weights.append(weight_layer.copy())

        for ind in xrange(-1, -len(layer_input_values) - 1, -1):
            current_inputs = layer_input_values[ind]
            current_weights = self.weight_layers[ind]
            if error_layer.shape[0] != 1:
                error_layer = error_layer.reshape(1, -1)
            if current_inputs.shape[1] != 1:
                current_inputs = current_inputs.reshape(-1, 1)
            updated_weights[ind] -= learning_rate * current_inputs.dot(error_layer)
            function_layer_index = ind - 1
            if function_layer_index < 0:
                continue
            # Update error layer
            error_layer = current_weights.dot(error_layer.T)
            # Passing df
            for err_ind in xrange(0, error_layer.size):
                error_val = error_layer[err_ind]
                cell = self.layers[function_layer_index][err_ind]
                result = cell.safe_df(error_val)
                result = np.squeeze(result)
                error_layer[err_ind] = result

        self.weight_layers = updated_weights
            
        

    def fit(self, x, y):
        self.weight_layers = []
        x = np.array(x)
        if len(x.shape) < 2:
            raise Exception('x must have more than 2 dims!')
        y = np.array(y)

        wlayer = np.ones((x.shape[1], len(self.layers[0])))
        self.weight_layers.append(wlayer)

        for ind in xrange(0, len(self.layers) - 1):
            layer = self.layers[ind]
            next_layer = self.layers[ind + 1]
            self.weight_layers.append(np.ones((len(layer), len(next_layer))))
            
        # Outupt layer
        self.weight_layers.append(np.ones((len(self.layers[-1]),
            y.shape[0])))
        
        output = self.forward_propergate(x)
        return output

    def predict(self, x, y):
        pass

    def show_graph(self):
        if len(self.weight_layers) == 0:
            print 'Empty model!'
            return None
        graph = nx.Graph()

        # NN layers
        pos_dict = dict()
        for layer_index in xrange(0, len(self.weight_layers)):
            weight_mat = self.weight_layers[layer_index]
            weight_shape = weight_mat.shape
            label_bias = 0.1
            for left_index in xrange(0, weight_shape[0]):
                for right_index in xrange(0, weight_shape[1]):
                    graph.add_edge('{}_{}'.format(layer_index, left_index),
                            '{}_{}'.format(layer_index + 1, right_index),
                            w = weight_mat[left_index, right_index])
                    pos_dict['{}_{}'.format(layer_index,
                        left_index)] = (layer_index, left_index)
                    pos_dict['{}_{}'.format(layer_index + 1,
                        right_index)] = (layer_index + 1, right_index)
                # nx.draw_networkx_edge_labels(graph,
                        # label_pos = label_bias,
                        # alpha = 0.1,
                        # font_size = 14,
                        # pos=pos_dict,
                        # nodecolor='r',
                        # edge_color='b')
                # label_bias += 0.8 / (weight_shape[0] + 1)
                # Remove edges in this layer
                # for right_index in xrange(0, weight_shape[1]):
                    # graph.remove_edge('{}_{}'.format(layer_index, left_index),
                            # '{}_{}'.format(layer_index + 1, right_index))
            
        nx.draw_networkx(graph,
                alpha = 0.5,
                pos=pos_dict,
                font_size = 8,
                nodecolor='r',
                edge_color='b',
                style = 'dotted')
        nx.draw_networkx_edge_labels(graph,
                alpha = 0.1,
                font_size = 14,
                label_pos = 0.6,
                pos=pos_dict,
                nodecolor='r',
                edge_color='b')
        plt.show()

def getSampleData():
    # x = np.array([[1,],[23,]])
    x = 121
    x = np.array(x).reshape(1,-1)
    print 'Data x.shape:', x.shape
    y = 5000
    return (x, y)

def Test():
    x, y = getSampleData()
    y = np.array(y).reshape(1,1)
    snn = SimpleNeuralNetwork(layers_config = ((10, 'relu'),(5, 'relu')))
    output, layer_inputs = snn.fit(x, y)
    print 'output  =', output
    print 'y = ', y
    print 'layer inputs:', layer_inputs
    for x_in in layer_inputs:
        print x_in.T
    for ind in xrange(0, 100):
        snn.back_propergate(x, y)
        output, layer_inputs = snn.forward_propergate(x)
        print 'y˙ = {}, y = {}'.format(output, y)
        snn.show_graph()
    
    
if __name__ == '__main__':
    Test()
