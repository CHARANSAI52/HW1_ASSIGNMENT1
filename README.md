# Neural Network & Deep Learning Assignment

## Student Information
- Student Name: PUPPALA CHARAN SAI
- Student ID: 700787150
- Course: CS5720 Neural network and Deep learning 
- Instructor:  I Hua Tsai
- Date: 09-09-2026

## Overview
This project contains solutions for:
1. Short-answer questions covering AI, ML, neural networks, perceptrons, activation functions, and training.
2. TensorFlow tensor manipulation and broadcasting.
3. MSE and categorical cross-entropy comparison.
4. MNIST training with Adam and SGD.
5. MNIST training with TensorBoard logging.

## Project Structure
```text
HW1_ASSIGNMENT1/
├── assignment.py
├── requirements.txt
├── README.md
├── plots/
└── logs/
    └── fit/
```

## Installation
```bash
python -m venv .venv
```

Activate the environment and install dependencies:
```bash
pip install -r requirements.txt
```

## Run
```bash
python assignment.py
```

The program creates plots in the `plots/` directory and TensorBoard event files in `logs/fit/`.

## TensorBoard
After training, run:
```bash
tensorboard --logdir logs/fit
```

Then open the local address shown by TensorBoard in a browser.

## GitHub Submission
1. Create a GitHub repository.
2. Copy this project into the repository.
3. student-information 
4. Commit and push:
```bash
git add .
git commit -m "Complete neural network assignment"
git push origin main
```
5. Submit the GitHub repository link and required video through Brightspace.

## Code Commenting
The Python code contains comments for the major operations, including preprocessing, tensor transformations, loss calculations, model construction, optimizer training, and TensorBoard logging.

## Part I - Short Answers

### Question 1

**a. Traditional programming vs. machine learning**

In traditional programming, the programmer explicitly writes the rules or logic that transform inputs into outputs. In machine learning, the programmer provides data and a learning algorithm, and the model learns patterns or parameters from that data to make predictions.

**b. Relationship among AI, ML, and DL**

Artificial Intelligence (AI) is the broad field of building systems that perform tasks associated with intelligent behavior. Machine Learning (ML) is a subset of AI in which systems learn patterns from data instead of relying only on explicitly programmed rules. Deep Learning (DL) is a subset of ML that uses neural networks with multiple layers to learn increasingly complex representations from data.

**c. Two reasons deep learning has become more successful**

1. Large datasets have become widely available, giving deep-learning models more examples from which to learn.
2. Modern GPUs/TPUs and improved hardware/software make it practical to train large neural networks much faster than in the past.

### Question 2

**a. Input, hidden, and output layers**

- The **input layer** receives the input features, such as pixel values from an image.
- The **hidden layers** transform the inputs through weighted calculations and activation functions to learn useful patterns and representations.
- The **output layer** produces the final prediction, such as class probabilities for a classification problem.

**b. Weights and biases**

Weights determine how strongly individual input values influence a neuron. A bias is an additional trainable value that shifts the neuron's weighted sum. An artificial neuron typically computes:

`z = w1*x1 + w2*x2 + ... + wn*xn + b`

The result `z` is then passed through an activation function.

**c. Why an activation function is needed**

An activation function introduces nonlinearity into a neural network. Without nonlinear activation functions, stacking multiple layers would still produce an overall linear transformation, limiting the network's ability to learn complex patterns such as those required for image and speech recognition.

### Question 3

**a. Perceptron and binary output**

A perceptron is a simple artificial neuron used for binary classification. It calculates a weighted sum of its inputs plus a bias and applies a threshold/step function. For example, it can output `1` when the weighted sum is above a threshold and `0` otherwise.

**b. Why one perceptron can solve AND and OR**

AND and OR are linearly separable problems. A single straight decision boundary can separate their positive examples from their negative examples, so one perceptron can learn suitable weights and bias values.

**c. Why one perceptron cannot solve XOR**

XOR is not linearly separable. No single straight decision boundary can correctly separate the XOR classes. A multilayer neural network solves this by combining multiple neurons and nonlinear activation functions, allowing it to form a nonlinear decision boundary.

### Question 4

**a. Sigmoid, Tanh, and ReLU**

- **Sigmoid:** outputs values from approximately `0` to `1`. It is useful for binary probabilities but can saturate at both ends.
- **Tanh:** outputs values from `-1` to `1`. It is zero-centered but can also saturate.
- **ReLU:** computes `max(0, x)`, so its output range is `[0, infinity)`. It is simple and usually provides stronger gradients for positive inputs.

**b. Vanishing gradients and ReLU**

The vanishing-gradient problem occurs when gradients become extremely small during backpropagation, especially through many layers. When this happens, early layers learn very slowly. Sigmoid and tanh can produce very small derivatives when they saturate. ReLU has a derivative of approximately `1` for positive inputs, so it generally helps preserve useful gradients in that region.

**c. Neural-network training cycle**

1. **Forward propagation:** Inputs pass through the network to produce predictions.
2. **Loss calculation:** The predictions are compared with the true labels using a loss function.
3. **Backpropagation:** The gradient of the loss with respect to the model parameters is calculated using the chain rule.
4. **Weight update:** An optimizer uses the gradients to update weights and biases so that future predictions should have lower loss.

## TensorFlow Broadcasting Explanation

Broadcasting allows TensorFlow to perform element-wise operations on tensors with compatible but different shapes without explicitly copying the smaller tensor. Dimensions are compatible when they are equal or when one of them is `1`. In this assignment, a tensor with shape `(1, 4)` can be added to a tensor with shape `(3, 2, 4)` because the `1` dimension can expand to match `3` and the last dimension `4` already matches.

## TensorBoard Questions

**1. What patterns do you observe in the training and validation accuracy curves?**

Typically, both training and validation accuracy increase during the early epochs. If they remain relatively close, the model is generalizing reasonably well. If training accuracy continues increasing while validation accuracy stops improving or begins decreasing, this indicates possible overfitting.

**2. How can TensorBoard detect overfitting?**

TensorBoard can display training and validation loss/accuracy on the same graphs. A common sign of overfitting is training loss continuing to decrease while validation loss begins to increase, or training accuracy continuing to rise while validation accuracy levels off or falls.

**3. What happens when you increase the number of epochs?**

Increasing epochs gives the model more opportunities to learn from the training data. Initially this can improve both training and validation performance. After enough epochs, however, validation performance may stop improving and overfitting may occur. Techniques such as early stopping, regularization, dropout, or more data can help control this.
