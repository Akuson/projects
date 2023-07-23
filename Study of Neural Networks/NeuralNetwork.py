import numpy as np

class NN(object):
    # len(layer_sizes) == layers + 1 ; (Since, input layer is also included.)
    def __init__(model, layers, layer_sizes, activations, loss_func, deriv_loss, method='standard',AdaM_opt=False,dropout=False):
        def assign_func(name:str):
            match name:
                case 'ReLU':
                    return (lambda x: x*(x>0), lambda y: 1*(y>0))
                case 'Leaky_ReLU':
                    return (lambda x: x*(x>0) + 0.5*x*(x<0), lambda y: 1*(y>0) + 0.5*(y<0))
                case 'Sigmoid':
                    return (lambda x: np.where(x<0,np.exp(x)/(1+np.exp(x)),1/(1+np.exp(-x))), lambda y: y*(1-y))
                case 'tanh':
                    return (lambda x: np.tanh(x),lambda y: 1 - (y**2))
                case 'my_transform': # With the unbound nature of 'ReLU' and the smooth nature of 'tanh'.
                    return(lambda x:np.log(np.abs((2*(x>0)-1)+x)), lambda y: 1/np.exp(np.abs(y)))
                case _:
                    return (lambda x: x, lambda y: 1)

        def set_hyper_params(model):
            # alpha = int(input("Enter the learning rate : "))
            model.alpha = 0.01
            # lmbd = int(input("Regularization rate (0 to nullify): "))
            model.lmbd = 0
            model.deriv_reglr = {"L2":lambda x: x, "L1":lambda x: 1}

            if(model.node_type=='batch_norm'):
                model.BN_decay = 0.9 # The exponentially weighted average decay factor of batch 'means' and 'stddevs'
            
            if(model.isAdaM):
                # b1 = int(input("Enter the first moment decay factor : "))
                b1 = 0.9
                # b2 = int(input("Enter the second moment decay factor : "))
                b2 = 0.999
                eps = 1.0e-8 # Numerical stability constant
                model.AdaM_params = {"b1":b1, "b2":b2, "eps":eps}
  
        model.cache = dict()
        
        model.back_trials = 0
        model.frwrd_trials = 0

        model.loss = loss_func
        model.deriv_loss = deriv_loss

        model.layers = np.asarray(layer_sizes,dtype=np.dtype("int64"))
        model.node_type = method
        model.isAdaM = AdaM_opt
        
        # This does not ensure that ONLY 80% of nodes will be dropped out. Maybe more, maybe less.
        model.dropout_probs = np.ones(shape=(layers+1, 1)) - (dropout)*np.random.random(size=(layers+1, 1)) / 5 # Not less than 80%
        model.dropout_probs[layers, 0] = model.dropout_probs[0, 0] = 1 # No dropout in input and output layers.

        # Declare and Define the model:
        # 'object' dtype as the different layers are of different dimensions.
        model.W_layers = np.empty((layers+1,),dtype='object')
        model.layer_activs = np.empty((layers+1,),dtype='object')
        model.layer_activ_derivs = np.empty((layers+1,),dtype='object')

        if(method=='standard'):
            model.b_layers = np.empty((layers+1,),dtype='object')
        elif(method=='batch_norm'):
            # Expected input feature distribution parameters of layers.
            model.E_stddev_layers = np.empty((layers+1,),dtype='object')
            model.E_mean_layers = np.empty((layers+1,),dtype='object')
            # Preferred/Trained ouput distribution parameters of layers.
            model.sig_layers = np.empty((layers+1,),dtype='object')
            model.mu_layers = np.empty((layers+1,),dtype='object')

        if(AdaM_opt):
            model.mnt_layers = np.empty((layers+1,),dtype='object')
            model.mnt_sqr_layers = np.empty((layers+1,),dtype='object')
        
        # Initialize the model:
        for l in range(1,layers+1):
            model.W_layers[l] = np.random.rand(layer_sizes[l],layer_sizes[l-1])
            model.layer_activs[l], model.layer_activ_derivs[l] = assign_func(activations[l-1])
            
            if(AdaM_opt):
                mnt_W = np.zeros((layer_sizes[l],layer_sizes[l-1]))
                mnt_sqr_W = np.zeros((layer_sizes[l],layer_sizes[l-1]))
            
            if(method=='standard'):
                model.b_layers[l] = np.zeros(shape=(layer_sizes[l], 1))
                
                if(AdaM_opt):
                    mnt_b = np.zeros(shape=(layer_sizes[l], 1))
                    mnt_sqr_b = np.zeros(shape=(layer_sizes[l], 1))

                    model.mnt_layers[l] = (mnt_W, mnt_b)
                    model.mnt_sqr_layers[l] = (mnt_sqr_W, mnt_sqr_b)
            
            elif(method=='batch_norm'):
                model.sig_layers[l] = np.ones(shape=(layer_sizes[l], 1))
                model.mu_layers[l] = np.zeros(shape=(layer_sizes[l], 1))

                # Default assumed 'normal' values. Justification? Numerical stability for batch_size = 1.
                model.E_stddev_layers[l] = np.ones(shape=(layer_sizes[l], 1))
                model.E_mean_layers[l] = np.zeros(shape=(layer_sizes[l], 1))
                
                if(AdaM_opt):
                    mnt_sig = np.zeros(shape=(layer_sizes[l], 1))
                    mnt_sqr_sig = np.zeros(shape=(layer_sizes[l], 1))
                    mnt_mu = np.zeros(shape=(layer_sizes[l], 1))
                    mnt_sqr_mu = np.zeros(shape=(layer_sizes[l], 1))

                    model.mnt_layers[l] = (mnt_W, mnt_sig, mnt_mu)
                    model.mnt_sqr_layers[l] = (mnt_sqr_W, mnt_sqr_sig, mnt_sqr_mu)
        
        # Set model hyperparameters:
        set_hyper_params(model)

    # Batch Normalization:
    """
    Given, an ndarray of features, normalizes them by Z-transform and returns the results.
    NOTE: Normalizes the features along axis-1 (i.e. the columns).

    Z-transform is a 1-to-1 map of any variable 'x' with an arbitrary distribution in examples:

        Z-transform(x) = (x - mean) / (stddev) ; 
            
            mean - is the average value of variable 'x' in its distribution.
            stddev - is the standard deviation of variable 'x' in its distribution.
    """
    def batch_normalize(model,l):

        dpt_mask = model.cache[f"mask{l}"]
        feature_batches = model.cache[f"X{l}"]

        # Shapes: (layer_size, 1)
        batch_m = model.cache[f"batch_m{l}"] = np.mean(feature_batches,axis=1,keepdims=True)
        batch_std = model.cache[f"batch_std{l}"] = np.std(feature_batches,axis=1,keepdims=True)

        # Broadcast and transform.
        model.cache[f"Z{l}"] = (feature_batches - batch_m) / np.where(batch_std==0,np.inf,batch_std) # To prevent division by 0.
        model.cache[f"Z_hat{l}"] = (model.sig_layers[l]*model.cache[f"Z{l}"] + model.mu_layers[l]) * dpt_mask

        # These exponentially weighted parameters are a form of "moving"/"sliding" metrics 
        # which are calculated over a "window" of observations of any vast distribution. 
        # In our case, this "window" is a specific batch of example data.
        # These exponentially weighted metrics, to some extent approximate the true values.
        model.E_mean_layers[l][dpt_mask[:,0]] = (model.BN_decay*model.E_mean_layers[l] + (1-model.BN_decay)*batch_m)[dpt_mask[:,0]] # Update expected means of layer.
        model.E_stddev_layers[l][dpt_mask[:,0]] = (model.BN_decay*model.E_stddev_layers[l] + (1-model.BN_decay)*batch_std)[dpt_mask[:,0]] # Update expected stddevs of layer.

        # NOTE: Bias Correction factor to be applied after, if required. NUMERICALLY UNSTABLE WHEN LARGE NUMBER OF TRIALS OCCURED.
        model.E_stddev_layers[l][dpt_mask[:,0]] = (model.E_stddev_layers[l]/(1-model.BN_decay**model.frwrd_trials))[dpt_mask[:,0]]
        model.E_mean_layers[l][dpt_mask[:,0]] = (model.E_mean_layers[l]/(1-model.BN_decay**model.frwrd_trials))[dpt_mask[:,0]]

    # Normalises a feature values of a network layer, with *predetremined* 'means' and 'stddevs'. 
    def normalize(model,l):
        # Shape: (layer_size,1)
        model.cache[f"Z{l}"] = ((model.cache[f"X{l}"] - model.E_mean_layers[l]) / np.where(model.E_stddev_layers[l]==0,np.inf,model.E_stddev_layers[l])) # To prevent division by 0.
        model.cache[f"Z_hat{l}"] = (model.sig_layers[l]*model.cache[f"Z{l}"] + model.mu_layers[l]) * model.cache[f"mask{l}"]
    
    # Forward propagation
    """
    Takes the 'outputs' of a former layer of network and processes them through the current layer.
    It first applies the weights(W) and biases(b) of the current layer onto the 'outputs' of former layer, 
    then transforms these linear combinations by applying the activation function, to give final result.
    """
    def forward_prop_layer(model,l): # ReLU default.
        dpt_mask = model.cache[f"mask{l}"]
        
        if(model.node_type=='batch_norm'):
            model.cache[f"X{l}"] = np.matmul(model.W_layers[l], model.cache[f"A{l-1}"]) * dpt_mask # Shape: (layer_size, batch_size)
            
            if(model.cache[f"X{l}"].shape[1]!=1):
                NN.batch_normalize(model,l)
            else:
                NN.normalize(model,l)
            
            model.cache[f"A{l}"] = model.layer_activs[l](model.cache[f"Z_hat{l}"]) * dpt_mask
        
        if(model.node_type=='standard'):
            model.cache[f"X{l}"] = (model.W_layers[l].dot(model.cache[f"A{l-1}"]) + model.b_layers[l]) * dpt_mask # Shape: (layer_size, batch_size)
            model.cache[f"A{l}"] = model.layer_activs[l](model.cache[f"X{l}"]) * dpt_mask

    # Backward propagation and Gradient Descent
    # Question: To divide by 'batch_size' or to not?
    """
    Updates the weights and biases of a layer from the output loss-gradients of the layers after, 
    and returns the output loss-gradients of the layers before as well as the 
    output loss gradients of weights and biases of the current layer.
    """
    def back_prop_layer(model,l):
        batch_size = model.cache[f"X{l}"].shape[1]
        model.cache[f"dA{l}"] = model.cache[f"dA{l}"] * model.cache[f"mask{l}"]
        
        NN.back_prop_nodes(model,l)

        # The partial derivative of total loss w.r.t ouputs of the former layer.
        model.cache[f"dA{l-1}"] = model.W_layers[l].T.dot(model.cache[f"dX{l}"]) # Shape: (prev_layer_size, batch_size)

        model.cache[f"dW{l}"] = model.cache[f"dX{l}"].dot(model.cache[f"A{l-1}"].T) / batch_size # Shape: (layer_size, prev_layer_size)
        
        if(model.lmbd!=0):
            d_reg_W = model.deriv_reglr["L2"](model.W_layers[l]) / batch_size # Shape: (layer_size, prev_layer_size)
            model.cache[f"dW{l}"] = model.cache[f"dW{l}"] + (model.lmbd * d_reg_W * model.cache[f"mask{l}"])
            
            if(model.node_type=='batch_norm'):
                d_sig_W = model.deriv_reglr["L2"](model.sig_layers[l]) / batch_size # Shape: (layer_size, prev_layer_size)
                model.cache[f"dSig{l}"] = model.cache[f"dSig{l}"] + (model.lmbd * d_sig_W * model.cache[f"mask{l}"])

        if(model.node_type=='standard'):
            model.cache[f"dB{l}"] = np.sum(model.cache[f"dX{l}"],axis=1,keepdims=True) / batch_size # Shape: (layer_size,1)

        NN.grad_descent(model,l)

    def back_prop_nodes(model,l):
        batch_size = model.cache[f"X{l}"].shape[1]

        if(model.node_type=='batch_norm'):
            # Shapes: (layer_size,batch_size)
            model.cache[f"dZ_hat{l}"] = model.layer_activ_derivs[l](model.cache[f"A{l}"]) * model.cache[f"dA{l}"]
            model.cache[f"dZ{l}"] = model.cache[f"dZ_hat{l}"] * model.sig_layers[l]
            
            # Shapes: (layer_size, 1)
            model.cache[f"dMu{l}"] = np.sum(model.cache[f"dZ_hat{l}"],axis=1,keepdims=True) / batch_size
            model.cache[f"dSig{l}"] = np.sum(model.cache[f"dZ_hat{l}"] * model.cache[f"Z{l}"],axis=1,keepdims=True) / batch_size

            # Shape: (layer_size, batch_size)
            model.cache[f"dX{l}"] = ((1/batch_size) * model.sig_layers[l] / np.where(model.cache[f"batch_std{l}"]!=0,model.cache[f"batch_std{l}"],np.inf)) * (-model.cache[f"dSig{l}"]*model.cache[f"Z{l}"] + (batch_size)*model.cache[f"dZ_hat{l}"] - model.cache[f"dMu{l}"])
        
        if(model.node_type=='standard'):
            model.cache[f"dX{l}"] = model.layer_activ_derivs[l](model.cache[f"A{l}"]) * model.cache[f"dA{l}"] # Shape: (layer_size, batch_size)

    # Question: Should the residual moments for AdaM optimized gradient descent update parameters for dropped out nodes in network?
    # In following implementation, the moments start to decay if nodes are dropped out, and keep updating the model parameters regardless of drop-out.
    def grad_descent(model,l):
        if(model.node_type=='standard'): # Model parameters: 'W' and 'b'.
            if(model.isAdaM):
                # Unpacking the first and second moment terms of 'W' and 'b'
                m_W, m_b = model.mnt_layers[l]
                m_sqr_W, m_sqr_b = model.mnt_sqr_layers[l]

                b1 = model.AdaM_params["b1"]
                b2 = model.AdaM_params["b2"]
                eps = model.AdaM_params["eps"]
                
                trials = model.back_trials 
                
                # Update Exponentially weighted averages
                m_W = b1*m_W + (1-b1)*model.cache[f"dW{l}"]
                m_sqr_W = b2*m_sqr_W + (1-b2)*(model.cache[f"dW{l}"]**2)

                m_b = b1*m_b + (1-b1)*model.cache[f"dB{l}"]
                m_sqr_b = b2*m_sqr_b + (1-b2)*(model.cache[f"dB{l}"]**2)

                # Does this work?
                model.mnt_layers[l], model.mnt_sqr_layers[l] = (m_W, m_b), (m_sqr_W, m_sqr_b)
                
                # Bias correction and update formula:
                model.cache[f"dW{l}"] = (m_W / (1-b1**trials)) / (np.sqrt(m_sqr_W/(1-b2**trials)) + eps)
                model.cache[f"dB{l}"] = (m_b / (1-b1**trials)) / (np.sqrt(m_sqr_b/(1-b2**trials)) + eps)

            model.W_layers[l] = model.W_layers[l] - model.alpha * model.cache[f"dW{l}"]
            model.b_layers[l] = model.b_layers[l] - model.alpha * model.cache[f"dB{l}"]
        
        if(model.node_type=='batch_norm'):
            if(model.isAdaM):
                m_W, m_sig, m_mu = model.mnt_layers[l]
                m_sqr_W, m_sqr_sig, m_sqr_mu = model.mnt_sqr_layers[l]

                # Unpacking hyper parameters.
                b1 = model.AdaM_params["b1"]
                b2 = model.AdaM_params["b2"]
                eps = model.AdaM_params["eps"]

                trials = model.back_trials
                
                # Update Exponentially weighted averages
                m_W = b1*m_W + (1-b1)*model.cache[f"dW{l}"]
                m_sqr_W = b2*m_sqr_W + (1-b2)*(model.cache[f"dW{l}"]**2)

                m_sig = b1*m_sig + (1-b1)*model.cache[f"dSig{l}"]
                m_sqr_sig = b2*m_sqr_sig + (1-b2)*(model.cache[f"dSig{l}"]**2)

                m_mu = b1*m_mu + (1-b1)*model.cache[f"dMu{l}"]
                m_sqr_mu = b2*m_sqr_mu + (1-b2)*(model.cache[f"dMu{l}"]**2)

                # Does this work?
                model.mnt_layers[l], model.mnt_sqr_layers[l] = (m_W, m_sig, m_mu), (m_sqr_W, m_sqr_sig, m_sqr_mu)

                # Bias correction and update formula:
                model.cache[f"dW{l}"] = (m_W / (1-b1**trials)) / (np.sqrt(m_sqr_W/(1-b2**trials)) + eps)
                model.cache[f"dSig{l}"] = (m_sig / (1-b1**trials)) / (np.sqrt(m_sqr_sig/(1-b2**trials)) + eps)
                model.cache[f"dMu{l}"] = (m_mu / (1-b1**trials)) / (np.sqrt(m_sqr_mu/(1-b2**trials)) + eps) 

            model.W_layers[l] = model.W_layers[l] - model.alpha * model.cache[f"dW{l}"]
            model.sig_layers[l] = model.sig_layers[l] - model.alpha * model.cache[f"dSig{l}"]
            model.mu_layers[l] = model.mu_layers[l] - model.alpha * model.cache[f"dMu{l}"]

    # Training routine
    def training_cycle(model, X, Y):
        
        layers = model.layers.size - 1

        # Q: Dropout, Batch Normalize, Transform?
        model.cache[f"mask{0}"] = (np.random.random(size=(model.layers[0], 1)) < model.dropout_probs[0, 0])
        model.cache[f"A{0}"] = X

        model.frwrd_trials += 1
        # Forward propagation through network.
        for l in range(1,layers+1):
            model.cache[f"mask{l}"] = (np.random.random(size=(model.layers[l], 1)) < model.dropout_probs[l, 0])
            NN.forward_prop_layer(model,l)

        # Loss calculations:
        Y_hat = model.cache[f"A{layers}"]
        total_loss = (1/X.shape[1]) * np.sum(model.loss(Y,Y_hat),axis=1)
        
        model.cache[f"dA{layers}"] = model.deriv_loss(Y,Y_hat)
        
        model.back_trials += 1
        # Back propagation through network.
        for l in range(layers,0,-1):
            NN.back_prop_layer(model,l)

        return total_loss

    # Inference routine
    def model_predict(model, X):

        layers = model.layers.size - 1

        model.frwrd_trials += 1
        model.cache[f"A{0}"] = X
        # Forward propagation through network.
        for l in range(1,layers+1):
            model.cache[f"mask{l}"] = (np.random.random(size=(model.layers[l], 1)) < 1)
            NN.forward_prop_layer(model,l)

        return model.cache[f"A{layers}"]
