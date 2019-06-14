import torch
import math

# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """  
    c = 0
    out=torch.zeros(outputs.size())
    for i in labels:
        out[c][i] = 1.
        c = c+1
    N=outputs.size(0)
    creloss = -torch.sum(out * torch.log(outputs))/N
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    k = 0
    N=outputs.size(0)
    out = torch.zeros(outputs.size()) 
    for j in labels:
        out[k][j] = 1.
        k+=1
    avg_grads = (outputs - out)/N
    return avg_grads

if __name__ == "__main__":
    pass
