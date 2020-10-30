import numpy as np
from scipy.special import softmax

class HiddenLayer:
    def __init__(self, input_size = 4, hidden_size = 3, output_size = 4):
        self.name = 'HiddenLayer'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_block = np.zeros(input_size)
        self.hidden_block = np.zeros(hidden_size)
        self.output_block = np.zeros(output_size)

        self.hidden_activation = 'relu'
        self.output_activation = 'sigmoid'

        self.reset_history()

    def initialize_weight(self, weight_type = -1, hidden_weight = None, output_weight = None, hidden_transition_weight = None):
        if weight_type == -1:
            # Random
            max_val = 1
            min_val = -1
            self.hidden_weight = (np.random.rand(self.input_size, self.hidden_size) * (max_val - min_val)) + min_val
            self.output_weight = (np.random.rand(self.hidden_size, self.output_size) * (max_val - min_val)) + min_val
            self.hidden_transition_weight = (np.random.rand(self.hidden_size, self.hidden_size) * (max_val - min_val)) + min_val
        elif weight_type == 0:
            # Zero
            self.hidden_weight = np.zeros((self.input_size, self.hidden_size))
            self.output_weight = np.zeros((self.hidden_size, self.output_size))
            self.hidden_transition_weight = np.zeros((self.hidden_size, self.hidden_size))
        elif weight_type == 1:
            # One
            self.hidden_weight = np.ones((self.input_size, self.hidden_size))
            self.output_weight = np.ones((self.hidden_size, self.output_size))
            self.hidden_transition_weight = np.ones((self.hidden_size, self.hidden_size))
        else:
            # Explicit
            self.hidden_weight = hidden_weight
            self.output_weight = output_weight
            self.hidden_transition_weight = hidden_transition_weight

    def initialize_bias(self, bias_type = -1, hidden_bias = None, output_bias = None):
        if bias_type == -1:
            # Random
            max_val = 1
            min_val = -1
            self.hidden_bias = (np.random.rand(self.hidden_size) * (max_val - min_val)) + min_val
            self.output_bias = (np.random.rand(self.output_size) * (max_val - min_val)) + min_val
        elif bias_type == 0:
            # Zero
            self.hidden_bias = np.zeros((self.hidden_size))
            self.output_bias = np.zeros((self.output_size))
        elif bias_type == 1:
            # One
            self.hidden_bias = np.ones((self.hidden_size))
            self.output_bias = np.ones((self.output_size))
        else:
            # Explicit
            self.hidden_bias = hidden_bias
            self.output_bias = output_bias

    def reset_history(self):
        self.input_history = []
        self.hidden_history = []
        self.output_history = []

    def get_history(self, layer_part = 'input', iteration = None):
        if layer_part == 'input':
            if iteration is None:
                return self.input_history
            else:
                if len(self.input_history) > iteration and iteration >= 0:
                    return self.input_history[iteration]
                else:
                    return 'ERROR. Iteration number is not correct'
        elif layer_part == 'hidden':
            if iteration is None:
                return self.hidden_history
            else:
                if len(self.hidden_history) > iteration and iteration >= 0:
                    return self.hidden_history[iteration]
                else:
                    return 'ERROR. Iteration number is not correct'
        elif layer_part == 'output':
            if iteration is None:
                return self.output_history
            else:
                if len(self.output_history) > iteration and iteration >= 0:
                    return self.output_history[iteration]
                else:
                    return 'ERROR. Iteration number is not correct'

    def forward_propagation_sequences(self, sequences, record_history = False):
        self.reset_history()
        self.previous_hidden = np.zeros(self.hidden_size)
        for sequence in sequences:
            self.forward_propagation(sequence, record_history)

    def forward_propagation(self, input_features, record_history = False):
        if record_history:
            self.input_history.append(input_features)

        sum_input = np.matmul(np.array(input_features), self.hidden_weight)
        sum_hidden = np.matmul(self.previous_hidden, self.hidden_transition_weight)

        hidden_total = sum_input + sum_hidden + self.hidden_bias

        self.previous_hidden = self.activation_function(hidden_total, self.hidden_activation)

        if record_history:
            self.hidden_history.append(self.previous_hidden)

        output_total = np.matmul(self.previous_hidden, self.output_weight) + self.output_bias

        self.previous_output = self.activation_function(output_total, self.output_activation)

        if record_history:
            self.output_history.append(self.previous_output)


    def activation_function(self, X, function_type):
        if function_type == 'sigmoid':
            X = np.clip(X, -500, 500)
            return 1.0/(1.0 + np.exp(-X))
        elif function_type == 'tanh':
            return np.tanh(X)
        elif function_type == 'relu':
            return np.maximum(0,X)
        elif function_type == 'softmax':
            return softmax(X)
        else:
            return X


if __name__ == '__main__':
    h1 = HiddenLayer()
    h1.initialize_weight(1)
    h1.initialize_bias(1)

    sequences = [[1, 2, -3, 4], [3, -1, 2, 0], [1, -1, 4, -3], [3, 2, 0, -1]]
    h1.forward_propagation_sequences(sequences, True)
    print(h1.input_history)
    print(h1.hidden_history)
    print(h1.output_history)