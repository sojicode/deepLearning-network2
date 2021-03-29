## DeepLearning - Network2

The original NNDL code "network2.py" is hard-coding several things, including the sigmoid activation function, L2 regularization and the input data format (for MNIST). In this assignment, we make the code general so that it (implements and) accepts various hyper-parameters as (literally) parameters of the network.

### Modification

1. Hyper-parameters:
> * Cost function: QuadraticCost, CrossEntropy, LogLikelihood 
> * Activation function: Sigmoid, Tanh, ReLU, Softmax 
> * Regularization: L1 and L2

2. Functions:
> * set_parameters()
> * feedforward() 
> * backprop() 
> * update_mini_batch() 
> * total_cost()

### Result
|  | act_hitten | act_output |  cost  | regularization | lmbda | dropout | result |
---|------------|------------|--------|----------------|-------|---------|--------|
1  | Sigmoid|Sigmoid|Quadratic|(default)| 0.0 | 0.0 | match
2  | Sigmoid|Sigmoid|CrossEntropy|(default)|0.0|0.0|match
3  | Sigmoid|Softmax|CrossEntropy|(default)|0.0|0.0|match
4  | Sigmoid|Softmax|LogLikelihood|(default)|0.0|0.0|match
5  | ReLU|Softmax|CrossEntropy|(default)|0.0|0.0|match
6  | ReLU|Softmax|LogLikelihood|(default)|0.0|0.0|match
7  | Tahn|Sigmoid|Quadratic|(default)|0.0|0.0|match
8  | Tahn|Tanh|Quadratic|(default)|0.0|0.0|match
9  | Sigmoid|Sigmoid|Quadratic|L2|3.0|0.0|match
10  | Sigmoid|Sigmoid|Quadratic|L1|3.0|0.0|match
11  | Sigmoid|Sigmoid|Quadratic|(default|0.0|0.1|Result may vary
12  | Sigmoid|Sigmoid|Quadratic|(default|0.0|0.5|Result may vary
