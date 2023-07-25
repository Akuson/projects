
A 2 Layer Neural Network to classify decimal digits : 0 to 9.

Input: A 28x28(784) grid pixels with integer brightness value [0,255] 

Uses Logistic Function at Output. Adaptive Moment Gradient Descent used during Training.

NN(
    layers : 2,
    layer_sizes: [784, 156, 10],
    activations: ["Sigmoid", "Sigmoid", ],
    loss_func: mse_loss,
    deriv_loss: mse_loss_deriv,
    method: 'batch_norm',
    AdaM_opt: True,
    dropout: True,
    )
