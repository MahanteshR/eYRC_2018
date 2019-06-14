import torch

# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors

    """
    result = 1/(1+torch.exp(-z))    #using torch.exp to calculate the exponential, sigmoid = 1/(1+exp(-x))  
    return result

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    """
    grad_sigmoid = sigmoid(z)*(1-sigmoid(z))       #.grad to find the gradient of the tensor.  
    return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

    """
    maxes = torch.max(x, 1, keepdim=True)[0]   
    masked_exp_xs = torch.exp(x)
    normalization_factor = masked_exp_xs.sum(1, keepdim=True)
    stable_softmax = masked_exp_xs / normalization_factor
    return stable_softmax

if __name__ == "__main__":
    pass
