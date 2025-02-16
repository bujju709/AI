from math import exp                   # For calculating the sigmoid activation function
from random import seed, random         # For initializing random weights

# Initialize a neural network
def initialize_network(n_inputs, n_hidden, n_outputs): 
    network = list()  # Create an empty list to store the layers of the network
    
    # Create the hidden layer
    # Each neuron has weights for each input + 1 weight for the bias
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] 
    network.append(hidden_layer)  # Add the hidden layer to the network
    
    # Create the output layer
    # Each output neuron has weights for each hidden neuron + 1 weight for the bias
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] 
    network.append(output_layer)  # Add the output layer to the network
    
    return network  # Return the initialized network

# Calculate the neuron's activation for an input
def activate(weights, inputs): 
    activation = weights[-1]  # Start with the bias (last weight)
    for i in range(len(weights) - 1):  # Loop through all weights except the bias
        activation += weights[i] * inputs[i]  # Sum the product of weights and inputs
    return activation 

# Sigmoid activation function
def transfer(activation): 
    return 1.0 / (1.0 + exp(-activation))  # Sigmoid function for non-linearity

# Forward propagate input to get the output of the network
def forward_propagate(network, row): 
    inputs = row  # Start with the input row
    for layer in network:  # Loop through each layer in the network
        new_inputs = []   # To store outputs of the current layer
        for neuron in layer:  # Loop through each neuron in the layer
            activation = activate(neuron['weights'], inputs)  # Calculate neuron activation
            neuron['output'] = transfer(activation)  # Apply activation function
            new_inputs.append(neuron['output'])  # Store the output for the next layer
        inputs = new_inputs  # Set the input for the next layer as the current layer's output
    return inputs  # Return the output of the last layer

# Calculate the derivative of neuron output for backpropagation
def transfer_derivative(output): 
    return output * (1.0 - output)  # Derivative of sigmoid function

# Backpropagate error and store it in neurons
def backward_propagate_error(network, expected): 
    # Loop from the last layer to the first layer
    for i in reversed(range(len(network))): 
        layer = network[i]  # Get the current layer
        errors = list()  # List to store error terms
        
        # If not the output layer, calculate the error from the next layer
        if i != len(network) - 1: 
            for j in range(len(layer)):  # Loop through each neuron in the layer
                error = 0.0
                # Calculate error as a sum of weighted errors from neurons in the next layer
                for neuron in network[i + 1]: 
                    error += (neuron['weights'][j] * neuron['delta']) 
                errors.append(error)  # Store the error
        else:
            # For output layer, calculate error as (expected - output)
            for j in range(len(layer)): 
                neuron = layer[j] 
                errors.append(expected[j] - neuron['output']) 
        
        # Calculate delta (error term) for each neuron
        for j in range(len(layer)): 
            neuron = layer[j] 
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output']) 

# Update network weights using error terms and learning rate
def update_weights(network, row, l_rate): 
    for i in range(len(network)): 
        inputs = row[:-1]  # Get inputs from the training row (excluding the target)
        if i != 0: 
            # For hidden and output layers, use the outputs from the previous layer as inputs
            inputs = [neuron['output'] for neuron in network[i - 1]] 
        
        # Update weights for each neuron
        for neuron in network[i]: 
            for j in range(len(inputs)): 
                # Update weight based on the input, error term (delta), and learning rate
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] 
            # Update bias (last weight)
            neuron['weights'][-1] += l_rate * neuron['delta'] 

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs): 
    for epoch in range(n_epoch):  # Loop over the number of epochs
        sum_error = 0  # To keep track of the total error
        
        # Loop over each row in the training dataset
        for row in train: 
            outputs = forward_propagate(network, row)  # Get the output from forward propagation
            
            # Create the expected output vector for the current row
            expected = [0 for i in range(n_outputs)] 
            expected[row[-1]] = 1  # Set the target class to 1 (one-hot encoding)
            
            # Calculate the sum of squared errors
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))]) 
            
            # Perform backpropagation and update weights
            backward_propagate_error(network, expected) 
            update_weights(network, row, l_rate) 
        
        # Print error for the current epoch
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error)) 

# Test the backpropagation algorithm with a small dataset
seed(1)  # Set seed for reproducibility
dataset = [
    [2.7810836, 2.550537003, 0], 
    [1.465489372, 2.362125076, 0], 
    [3.396561688, 4.400293529, 0], 
    [1.38807019, 1.850220317, 0], 
    [3.06407232, 3.005305973, 0], 
    [7.627531214, 2.759262235, 1], 
    [5.332441248, 2.088626775, 1], 
    [6.922596716, 1.77106367, 1], 
    [8.675418651, -0.242068655, 1], 
    [7.673756466, 3.508563011, 1]
] 

n_inputs = len(dataset[0]) - 1  # Number of inputs (excluding the target)
n_outputs = len(set([row[-1] for row in dataset]))  # Number of unique classes (0 or 1)

# Initialize the neural network
network = initialize_network(n_inputs, 2, n_outputs)  # 2 neurons in hidden layer
train_network(network, dataset, 0.5, 20, n_outputs)   # Train the network

# Print the learned weights for each layer
for layer in network: 
    print(layer)
