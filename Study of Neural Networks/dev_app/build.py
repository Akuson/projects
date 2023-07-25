import numpy as np
from NeuralNetwork import NN
from losses import mse_loss,mse_loss_deriv,softmax_transform


def build_model():
    model = NN(2,[784, 156, 10],["Sigmoid", "Sigmoid", ],mse_loss,mse_loss_deriv,'batch_norm',True,True)

    path = "./95_model/95_model_"
    for l in range(1,model.layers.size):
        model.W_layers[l] = np.loadtxt(path+f"W_{l}.txt").reshape(model.layers[l], model.layers[l-1])
        
        if(model.node_type=='batch_norm'):
            model.sig_layers[l] = np.loadtxt(path+f"sig_{l}.txt").reshape((model.layers[l],1))
            model.mu_layers[l] = np.loadtxt(path+f"mu_{l}.txt").reshape((model.layers[l],1))
            model.E_stddev_layers[l] = np.loadtxt(path+f"stddev_{l}.txt").reshape((model.layers[l],1))
            model.E_mean_layers[l] = np.loadtxt(path+f"mean_{l}.txt").reshape((model.layers[l],1))
        if(model.node_type=='standard'):
            np.savetxt(path+f"b_{l}.txt",model.b_layers[l])

    return model

def predict_digit(model:NN,image_data):
    image_data = np.asarray(image_data).flatten().reshape((784,1))
    pred = model.predict(image_data)
    return np.argmax(pred)
    # return softmax_transform(pred)