import numpy as np

def softmax_transform(z):
    # For numerical stability
    z = z - np.max(z,axis=0,keepdims=True)

    exp_transform = np.exp(z)
    return exp_transform / np.sum(exp_transform,axis=0)
# It is a measure of accuracy: Higher means more accurate.
def softmax_loss(y,z):
    # y_hat = softmax_transform(z)
    # return np.sum(-(y*np.log(y_hat)),axis=0,keepdims=True)
    return np.sum((y * softmax_transform(z)) > 0.5,axis=0,keepdims=True)
def softmax_loss_deriv(y,z):
    # -(y/y_hat) is the partial derivative of loss w.r.t 'softmax' output.
    
    # **This is the partial derivative of loss w.r.t 'softmax' input.
    # **Under the assumption**: For each of the labels(y) in data, there is one class which is '1' and all others are '0'.(i.e. 1-Hot Encoded Labels)  
    return (softmax_transform(z) - y)

# Another function whose derivative is completely defined by its output! Such functions seem to appear more often than I would expect.
def softmax_deriv(y_hat): # 'y_hat' has shape: (output_classes, batch_size)
    y_hat = y_hat.T

    # The (output_classes, output_classes) partial derivative matrix is symmetric for each example.
    derivs = -np.matmul(np.reshape(y_hat,(y_hat.shape[0], y_hat.shape[1], 1)), np.reshape(y_hat,(y_hat.shape[0], 1, y_hat.shape[1]))) 
    derivs = derivs + np.identity(derivs.shape[1])*(y_hat)

    # A 3-D grid of shape: (batch_size, output_classes, output_classes, )
    # Index (i,j,k) represents the partial derivative of 's-j' w.r.t. 'z-k' for the i'th example.
    # where, Z is the input vector to 'Softmax' function, and S is the output vector.
    return derivs

def mse_loss(y,y_hat):
    return (1/y.shape[0])*(y_hat-y)**2
def mse_loss_deriv(y,y_hat):
    return 2 * (y_hat - y) / y.shape[0]

def log_loss(y,y_hat):
    return -np.sum(y*np.log(y_hat)-(1-y)*np.log(1-y_hat),axis=0,keepdims=True)
def log_loss_deriv(y,y_hat):
    return (1-y)/(1-y_hat) - y/y_hat