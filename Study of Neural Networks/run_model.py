import numpy as np
from NeuralNetwork import NN
from losses import mse_loss,mse_loss_deriv

model = NN(2,[784, 156, 10],["Sigmoid", "Sigmoid", ],mse_loss,mse_loss_deriv,'batch_norm',True,True)

path = "./model/model_"
for l in range(1,model.layers.size):
    model.W_layers[l] = np.loadtxt(path+f"W_{l}.txt")
    model.sig_layers[l] = np.loadtxt(path+f"sig_{l}.txt")
    model.mu_layers[l] = np.loadtxt(path+f"mu_{l}.txt")
    model.E_stddev_layers[l] = np.loadtxt(path+f"stddev_{l}.txt")
    model.E_mean_layers[l] = np.loadtxt(path+f"mean_{l}.txt")


