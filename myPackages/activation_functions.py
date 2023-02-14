import numpy as np
import math
from scipy.special import softmax

ALPHA = 0.001

def linear_activation(input: float) -> float:
    # linear function
    return input

def deriv_linear(output: float) -> float:
    return 0

def hyperbolic_activation(input: float) -> float:
    # hyperbolic function, i.e. tanh()
    return math.tanh(input)

def deriv_hyperbolic(output: float) -> float:
    return 1-output**2

def sigmoid_activation(input: float) -> float:
    #sigmoid activation function
    return 1/(1+math.exp(-input))

def deriv_sigmoid(output: float) -> float:
    return output * (1- output)

def relu_activation(input: float) -> float:
    if input < 0:
        return 0
    else:
        return input
    
def deriv_relu(output: float) -> float:
    if output < 0:
        return 0
    else:
        return 1
    
def prelu_activation(input: float) -> float:
    if input < 0:
        return ALPHA*input
    else:
        return input
    
def deriv_prelu(output: float) -> float:
    if output < 0:
        return ALPHA
    else:
        return 1
    
def elu_activation(input: float) -> float:
    if input < 0:
        return ALPHA*(math.exp(input)-1)
    else:
        return input
    
def deriv_elu(output: float) -> float:
    if output < 0:
        return output + ALPHA
    else:
        return 1
    
def softmax_activation(input: list) -> list:
    return softmax(input)
