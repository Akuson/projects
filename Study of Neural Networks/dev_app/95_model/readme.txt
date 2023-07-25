""" 
The model that got me to 95% accuracy and its training journey:

    model_A = NN(2,[input_dims, input_dims//5, output_dims],["Sigmoid", "Sigmoid", ],mse_loss,mse_loss_deriv,'batch_norm',True,True)

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

Before this final phase of training we were teaching the model to minimise loss, which translates to being able to tell EXACTLY what a 0 or 1 or ... etc. is.
This is reflected in the fact the labels, which are 1-Hot encoded. However, this is not what our purpose of learning is. We wish to teach the model to tell apart 
a 1 from a 0, or a 0 from a 6, not to teach it to chase perfect definitions that the labels assume, but might not even exist in reality. Hence, in the final phase, 
when we focus the model to a small fraction of mis-labelled data only, and not let minor deviations from the majority of the predictions guide its learning, the model 
is able to fulfill its true purpose of being able to tell the symbols apart, instead of focusing on the most prominent distinctions only to minimise loss. 

The labels are a misrepresentation of the data, and the reality in the relations that allow us to classify a digit as a 0 or a 1 etc., not because it is nothing 
like any other symbol in existence(1-Hot Nature), but because it is something like a symbol that we have seen before, maybe fractionally more so than others.

Focus more on what it is, than on what it isn't, for often due to lack of comparisons with the negatives, we misjudge how different two things really are.

Since we have a finite number of neurons, we would rather some neurons forget what makes a 4 different from a 9, thus increasing the probabilities of each for the other, 
hence increasing apparent loss in the majority of 9's and 4's, but instead learn what makes the minority of slanted 4, similar to the straight 4, allowing the slanted 4 to have a higher probability of being 4, despite 
the net increase in loss due to lesser sureity for the majority outweights the decrease in loss for the minority of exceptions. To allow this we simply **try to** omit the majority from the loss term.

When trying to reproduce the same result, got stuck at 94% accuracy. Then made the batch size(0.02) and total_trials(21) even smaller, and removed the condition for ommiting batches.
This seemed to help. But why? Before this another unintuitive thing ocurred. I switched loss to softmax at about 94.1% and then trained on it for some time. After this, when I switched 
back to logistic loss, it was showing negative. Don't understand this interaction between the two, when applied to the model.


Can digits be recurrent data, as a sequence of frames, from the moment the pen is placed to the moment it is lifted?
"""