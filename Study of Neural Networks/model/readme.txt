""" 
MNIST Handwritten Decimal Digits (Input: 28x28 pixel grid with brightness values [0,255] ; Labels: 0-9)

The model that got me to 95% accuracy and its training journey:

    model_A = NN(2,[784, 156, 10],["Sigmoid", "Sigmoid", ],mse_loss,mse_loss_deriv,'batch_norm',True,True)

Started with MSE loss, then shifted to softmax loss(but I think that was bad decision as data was not sharp and many samples were confusing) 
then finally settled with logistic loss.

Initially trained with Large dropout at high learning rate with mini-batches with MSE.

Once saturated, shifted to log loss training with lower learning rates with no dropout.

Continued training till saturated log loss.

Then started to train on specific batches with high loss thershold(~0.105) while adjusting alpha, total_trials for each batch, and also the sample_ratios intuitively.

    Usually the larger the sample_ratio, the lesser the total_trials dedicated to train it.

    Counterintuitively, during this last phase of training, the sum of weights and biases, kept changing alot while still improving accuracy score, 
    even though the training was done at exponentially lower learning rates. Maybe the space is very sharp and ragged, so the model was unable to traverse 
    it well earlier. One of the factors that caused this is the capabilites of Adaptive Moment Gradient Descent.

Reached 95%.
"""