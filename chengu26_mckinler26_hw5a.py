import numpy as np
import random

# CS 421 - Homework 5: Part A
# Date: Nov. 4, 2025
# Authors: Malissa Chen and Rhiannon McKinley

# step 1:
# network structure
n_input = 4
n_hidden = 8
n_output = 1

# initialize weights
np.random.seed(0)
w_hidden = np.random.uniform(-1.0, 1.0, (n_hidden, n_input + 1))  # +1 for bias
w_output = np.random.uniform(-1.0, 1.0, (n_output, n_hidden + 1)) # +1 for bias

# step 2:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_matrix(x, w_hidden, w_output):

    # Do one forward pass through the neural network.
    # x: input array of 4 numbers 
    # w_hidden: weights from input -> hidden layer (8 rows x 5 cols)
    # w_output: weights from hidden -> output layer (1 row x 9 cols)

    # add bias input
    x_with_bias = np.append(x, 1)  # now x has 5 numbers

    # calculate hidden layer signals using matrix multiplication
    hidden_inputs = np.dot(w_hidden, x_with_bias)  

    # apply sigmoid to get hidden layer outputs 
    hidden_outputs = sigmoid(hidden_inputs)

    # add bias to hidden layer outputs (for the output layer’s bias)
    hidden_with_bias = np.append(hidden_outputs, 1)  # now has 9 numbers

    # Calculate final output
    final_input = np.dot(w_output, hidden_with_bias)  
    final_output = sigmoid(final_input)             

    return hidden_with_bias, final_output

# step 3:
# Cited from Copilot
def sigmoid_derivative(x):
    return x * (1 - x)

def backpropagation(x, target, W_hidden, W_output, learning_rate=0.1):
    # Forward pass
    h_b, y = forward_matrix(x, W_hidden, W_output)

    # Compute output error
    error_output = target - y  # shape (1,)
    delta_output = error_output * sigmoid_derivative(y)  # shape (1,)

    # Compute hidden layer error
    h = h_b[:-1]  # remove bias from hidden output
    W_output_no_bias = W_output[:, :-1]  # shape (1, 8)
    error_hidden = delta_output.dot(W_output_no_bias)  # shape (8,)
    delta_hidden = error_hidden * sigmoid_derivative(h)  # shape (8,)

    # Update output weights
    W_output += learning_rate * delta_output.reshape(-1, 1) * h_b.reshape(1, -1)

    # Prepare input with bias
    x_b = np.append(x, 1)  # shape (5,)
    # Update hidden weights
    W_hidden += learning_rate * delta_hidden.reshape(-1, 1) * x_b.reshape(1, -1)

    return W_hidden, W_output

# step 4:
# training data (from PartA_TrainingData.txt)
examples = [
    ([0, 0, 0, 0], [0]),
    ([0, 0, 0, 1], [1]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 1, 1], [1]),
    ([0, 1, 0, 0], [0]),
    ([0, 1, 0, 1], [1]),
    ([0, 1, 1, 0], [0]),
    ([0, 1, 1, 1], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([1, 0, 1, 0], [1]),
    ([1, 0, 1, 1], [1]),
    ([1, 1, 0, 0], [0]),
    ([1, 1, 0, 1], [0]),
    ([1, 1, 1, 0], [0]),
    ([1, 1, 1, 1], [1])
]

# training loop
max_epochs = 10000    
epoch = 0
average_error = 1.0

while average_error > 0.05 and epoch < max_epochs:
    total_error = 0.0
    # randomly pick 10 examples each epoch
    samples = random.sample(examples, 10)

    # train on each sample
    for x, target in samples:
        x = np.array(x)
        target = np.array(target)

        # forward + backprop update
        W_hidden, W_output = backpropagation(x, target, w_hidden, w_output, learning_rate=0.5)

        # calculate network output (to measure error)
        _, output = forward_matrix(x, W_hidden, W_output)
        sample_error = (target - output) ** 2
        total_error += sample_error

        # update global weights (so the network keeps improving)
        w_hidden, w_output = W_hidden, W_output

    # compute average error for this epoch
    average_error = total_error.mean()
    epoch += 1
    print(f"Epoch {epoch}: Avg Error = {average_error:.4f}")

print("\nTraining complete!")
print(f"Final average error: {average_error:.4f}")
