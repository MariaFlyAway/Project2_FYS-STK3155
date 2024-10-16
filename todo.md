# To Do:

## Coding

### a.

Implement gradient descent and stochastic gradient descent - *done, needs some fine tuning*

- Plain gradient descent 
- Gradient descent with momentum
- Stochastic gradient descent
- AdaGrad - with and without momentum for plain and SGD
- RMSProp
- ADAM

Analyze the convergence as a function of $\eta$ for OLS and Ridge, for Ridge also look at $\lambda$. *plot MSE as a function of learning rate for the different methods*. Look at MSE as function of number of epochs, size of batches.  
Use analytical gradient, then replace with autograd/JAX.  
Compare results with those from scikitlearns SGD options.

### b.

Write your own neural network - *done, need to implement backpropagation properly*  

- Look at the regression problem (use Franke function for example)
- Discuss choice of cost function
- Sigmoid function for activation function for hidden layers - which activation function for final layer?
- Train network and compare results with those form project 1
- Compare results against similar code using scikit-learn/tensorflow/pytorch
- Comment results and give a critical discussion of results obtained with linear regression code and own neural network
- make an analysis of the regularization parameters and the learning rates employed to find the optimal MSE and R2 scores

### c.

Test different activation functions for the hidden layers

- sigmoid
- ReLU
- leaky ReLU
Discuss results  
Can also study the way you initialize the weights and biases

### d. 

Perform a classification analysis on the Wisconsin breast cancer dataset using your neural net - *mostly done, already started work*  
Discuss your results and give a critical analysis of the various parameters, including hyper-parameters like the learning rate and $\lambda$, activation functions, number of hidden layers and nodes  
Compare results to that of scikit-learn/tensorflow/pytorch

### e.

Compare the FFNN code to Logistic regression
- write a logistic regression code using your SGD algorithm
- study results as a function of the hyperparameters
- add a L2 regularization parameter $\lambda$
- compare results to FFNN as well as scikit-learns logistic regression functionality


### f. 

Summarize the various algorithms and give a critical evaluation of their pros and cons  
Which algorithm works best for the regression and the classification case