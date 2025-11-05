import numpy as np
import random

# CS 421 - Homework 5: Part A
# Date: Nove. 4, 2025
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
