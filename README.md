# Deep Learning

This repository keeps record of Deep Learning Course (ID: 521153S-3003) from University of Oulu, Finland.

## Packages

- Python 3.8
- Pytorch 1.10

## Content

### 1. Linear regression: loss function, gradient descent, regularization

Problem setting: given the training set, how to train a model to predict the output of a new value of the input from the testing set?

<p align="center">
<img src="img/pic.jpg" width="400">
</p>

#### The solutions:

- **Step 1: define a set of functions**
  - Linear hypothesis:
  
      $$y = b + w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + ... + w_n * x_n$$
      
       - $w_1$, ..., $w_n$ are known as weights (can be any values)
      
       - $b$ is known as bias term (can be any value)
      
     In short, the hypothesis can be written as ( $w$, $x$ as vectors):  
     
     $$\hat{y} = f(x) = b +  w ^ T x$$

- **Step 2: Goodness of the functions**
  - Loss/cost function: We want a line having small residuals. --> minimize the sum of squared residuals (least squares method).
  
<p align="center">
<img src="img/loss.jpg" width="300">
</p>
    $$L(f) = L(w,b) = \frac{1}{2N} \sum_{i = 1}^{N} (y ^ i - (b + w ^ T x ^ i)) ^ 2$$
      
- **Step 3: Pick the best functions**
  - Minimize the cost (Gradient descent)
  
$${dL \over dw} = \frac{1}{N} \sum_{i = 1}^{N} (y ^ i - (b + w ^ T x ^ i))(-x ^ i) = 0 \implies w = {N \sum x ^ i y ^ i - \sum x ^ i \sum y ^ i \over N \sum x ^ i x ^ i - \sum x ^ i \sum x ^ i}$$

$${dL \over db} = \frac{1}{N} \sum_{i = 1}^{N} (y ^ i - (b + w ^ T x ^ i))(-1) = 0 \implies b = {\sum x ^ i x ^ i \sum y ^ i - \sum x ^ i y ^ i \sum x ^ i \over N \sum x ^ i x ^ i - \sum x ^ i \sum x ^ i}$$

Model selection: 

<p align="center">
<img src="img/model.jpg" width="400">
</p>

Model should be "simple" so that works on test data. To avoid overfitting $\implies$ introduce **Regularization** into loss function (back to step 2)

$$L(f) = L(w,b) = \frac{1}{2N} (\sum_{i = 1}^{N} (y ^ i - (b + w ^ T x ^ i)) ^ 2 + \sum_{i = 1}^{N} w ^ 2) $$



### 2. Linear Classification 

Examples: Pedestrian detection, Email spam detection, Tumor detection, Object classification

- Sigmoid $\implies$ binary classification methods where we only have 2 classes

- Softmax Classifer (Multinomial logistic regression) $\implies$ multiclass problems
  - The softmax activation function transforms the raw outputs of the neural network into a vector of probabilities, essentially a probability distribution over the input classes.

### 3. Neural Networks

- Neural Networks
- Multilayer Neural Networks
- Backpropogation
  - recursive application of the chain rule along a computational graph to compute the gradients of all inputs, parameters, and intermediates.
  - Forward: compute result of an operation and save any intermediates needed for gradient computation in memory
  - Backward: apply the chain rule to compute the gradient of the loss function with respect to the inputs

### 4. Convolutional Neural Network (CNN)



