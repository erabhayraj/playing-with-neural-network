import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return np.maximum(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (np.array(x)>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return 1/(1+np.exp(-x))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1-s)

## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return np.tanh(x)

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    return 1-(tanh(x)**2)

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        self.linear =[]
        self.nonlinear = []
        self.velocity_w=[]
        self.velocity_b=[]
        self.t=0
        self.second_moment_w=[]
        self.second_moment_b=[]
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        ## TODO 2: Initialize weights and biases for all hidden and output layers
        ## Initialization can be done with random normal values, you are free to use
        ## any other initialization technique.
        self.weights = []
        self.biases = []
        prev=input_dim
        for dims in hidden_dims:
            self.weights.append(np.random.normal(0,1,(dims,prev)))
            self.biases.append(np.random.normal(0,1,(dims,1)))
            prev=dims
        self.weights.append(np.random.normal(0,1,(1,prev)))
        self.biases.append(np.random.normal(0,1,(1,1)))

    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass

        ## TODO 3a: Compute activations for all the nodes with the corresponding
        ## activation function of each layer applied to the hidden nodes
        prev=np.transpose(X)
        self.linear=[]
        self.nonlinear=[]
        for i in range(0,len(self.weights)):
            z=np.dot(self.weights[i],prev) + self.biases[i]
            prev=self.activations[i](z)
            self.linear.append(z)
            self.nonlinear.append(prev)

        ## TODO 3b: Calculate the output probabilities of shape (N, 1) where N is number of examples
        output_probs=np.array(np.transpose(prev))
        return output_probs

    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation

        ## TODO 4a: Compute gradients for the output layer after computing derivative of 
        ## sigmoid-based binary cross-entropy loss
        ## Hint: When computing the derivative of the cross-entropy loss, don't forget to 
        ## divide the gradients by N (number of examples)  
        y=y.T
        ycap = self.nonlinear[-1]
        loss=-((y/ycap)-((1-y)/(1-ycap)))/X.shape[0]
        grad_weights_local=[]
        grad_biases_local=[]
        for i in reversed(range(len(self.weights))):
            dz=loss*self.activation_derivatives[i](self.linear[i])
            grad_biases_local.append(np.sum(dz,axis=1, keepdims=True))
            if i==0:
                grad_weights_local.append(np.dot(dz,X))
            else:
                grad_weights_local.append(np.dot(dz,self.nonlinear[i-1].T))
            loss=np.dot(self.weights[i].T,dz)

        ## TODO 4b: Next, compute gradients for all weights and biases for all layers
        ## Hint: Start from the output layer and move backwards to the first hidden layer
        self.grad_weights = grad_weights_local
        self.grad_biases = grad_biases_local
        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla GD, 2 for GD with Exponential Decay, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        ### Calculate updated weights using methods as indicated by gd_flag

        ## TODO 5a: Variant 1(gd_flag = 1): Vanilla GD with Static Learning Rate
        ## Use the hyperparameter learning_rate as the static learning rate
        updated_W,updated_B = [],[]
        n=len(self.weights)
        if gd_flag == 1:
            for i in range(len(weights)):
                updated_W.append(weights[i]-learning_rate*delta_weights[n-1-i])
                updated_B.append(biases[i]-learning_rate*delta_biases[n-1-i])

        ## TODO 5b: Variant 2(gd_flag = 2): Vanilla GD with Exponential Learning Rate Decay
        ## Use the hyperparameter learning_rate as the initial learning rate
        ## Use the parameter epoch for t
        ## Use the hyperparameter decay_constant as the decay constant
        elif gd_flag == 2:
            updated_learning_rate=learning_rate*(np.exp(-decay_constant*epoch))
            for i in range(len(weights)):
                updated_W.append(weights[i]-updated_learning_rate*delta_weights[n-1-i])
                updated_B.append(biases[i]-updated_learning_rate*delta_biases[n-1-i])


        ## TODO 5c: Variant 3(gd_flag = 3): GD with Momentum
        ## Use the hyperparameters learning_rate and momentum
        #vt = βvt−1 + (1 − β)∇w L(wt−1)
        elif gd_flag == 3:
            if epoch==0:
                self.velocity_w = [np.zeros_like(w) for w in weights]
                self.velocity_b = [np.zeros_like(b) for b in biases]
            for i in range(n):
                vt_w= momentum*self.velocity_w[i]+(1-momentum)*delta_weights[n-1-i]
                vt_b= momentum*self.velocity_b[i]+(1-momentum)*delta_biases[n-1-i]
                updated_W.append(weights[i]-learning_rate*vt_w)
                updated_B.append(biases[i]-learning_rate*vt_b)
                self.velocity_w[i]=vt_w
                self.velocity_b[i]=vt_b

        
        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta = optimizer_params['beta1']
        gamma = optimizer_params['beta2']
        eps = optimizer_params['eps']       

        ## TODO 6: Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer
        updated_W,updated_B = [],[]
        n=len(self.weights)
        if self.t==0:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]
            self.second_moment_w = [np.zeros_like(w) for w in weights]
            self.second_moment_b = [np.zeros_like(b) for b in biases]
        self.t+=1
        for i in range(n):
            vt_w= beta*self.velocity_w[i]+(1-beta)*delta_weights[n-1-i]
            vt_b= beta*self.velocity_b[i]+(1-beta)*delta_biases[n-1-i]
            st_w= gamma*self.second_moment_w[i]+((1-gamma)*(delta_weights[n-1-i]**2))
            st_b= gamma*self.second_moment_b[i]+((1-gamma)*(delta_biases[n-1-i]**2))
            scap_w=st_w/(1-gamma**self.t)
            scap_b=st_b/(1-gamma**self.t)
            vcap_w=vt_w/(1-beta**self.t)
            vcap_b=vt_b/(1-beta**self.t)
            offset_w=learning_rate/(np.sqrt(scap_w)+eps)
            offset_b=learning_rate/(np.sqrt(scap_b)+eps)
            updated_W.append(weights[i]-offset_w*vcap_w)
            updated_B.append(biases[i]-offset_b*vcap_b)
            self.velocity_w[i]=vt_w
            self.velocity_b[i]=vt_b
            self.second_moment_w[i]=st_w
            self.second_moment_b[i]=st_b
        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f'loss_bgd_' + str(optimizer_params['gd_flag']) + '.png')
        else:
            plt.savefig(f'loss_adam.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4,2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid']
    optimizer = "bgd"
    optimizer_params = {
        'learning_rate': 0.1,
        'gd_flag': 3,
        'momentum': 0.99,
        'decay_constant': 0.2
    }
    
    # For Adam optimizer you can use the following
    optimizer = "adam"
    optimizer_params = {
        'learning_rate': 0.01,
        'beta1' : 0.9,
        'beta2' : 0.999,
        'eps' : 1e-8
    }

     
    model = NN(input_dim, hidden_dims)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    test_preds = model.forward(X_eval)
    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)
