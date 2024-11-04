import numpy as np

import numpy

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1: 
# Instructions:
#Write a function that takes one numeric argument as input. 
#If the number is larger than zero, the function should return 1, otherwise is should return -1.
#The name of the function should be step

# Your code here:
# -----------------------------------------------

def step(x):
  if x > 0:
    return 1
  else:
    return -1

# -----------------------------------------------


# Task 2:
# Instructions:
#Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
#The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
#The name of the function should be ReLu


# Your code here:
# -----------------------------------------------
def ReLu(x, cutoff=0):
  return np.where(x < cutoff, cutoff, x)


# -----------------------------------------------


# Task 3:
# Instructions:
#Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
#The function should start by multiplying the two numpy arrays (matrix multiplication).
#Next, apply the ReLu function from above to the resulting matrix and return the result.
#Name the function neural_net_layer

# Your code here:
# -----------------------------------------------

def neural_net_layer(x, y):
  #using ndim to return an error when x isnt 2D and y isnt 1D
  #using shape to make sure that p is the same across x and y
  if x.ndim != 2 or y.ndim != 1 or x.shape[1] != y.shape[0]:
    raise ValueError("Wrong dimensions, x needs to be 2-dimensional and y needs to be 1-dimensional")
  #matrix multiplication
  w = x@y
  #running the result through ReLu
  return(ReLu(w))

# ------------------------------------------
