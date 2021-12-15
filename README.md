# Neural Networks MNIST Image Classifier 
This is a simple convolutional neural network to classify grayscale images of handwritten digits (0 - 9) from the MNIST dataset. 
The dataset contains 60,000 training images and 10,000 testing images of handwritten digits, 0 - 9. 
Each image is 28×28 pixels in size with only a single channel. 
It also includes labels for each example, a number indicating the actual digit (0 - 9) handwritten in that image.

The following shows some example images from the MNIST dataset:

![image](https://user-images.githubusercontent.com/59906096/146250413-a9c6b097-79b4-4b2f-87d7-99e99b28fbc9.png)

The starter code splits the set of 60,000 training images and labels into a sets of 50,000 examples as
the training set and 10,000 examples for dev set.
To start, you will implement a neural network with a single hidden layer and cross entropy loss, and
train it with the provided data set. Use the sigmoid function as activation for the hidden layer, and
softmax function for the output layer. Recall that for a single example (x, y), the cross entropy loss is:

![image](https://user-images.githubusercontent.com/59906096/146250850-6dc5e367-1d3d-4e51-8885-3f4d25f2be5f.png)

where ˆy ∈ R
K is the vector of softmax outputs from the model for the training example x, and y ∈ R
K
is the ground-truth vector for the training example x such that y = [0, ..., 0, 1, 0, ..., 0]> contains a
single 1 at the position of the correct class (also called a “one-hot” representation).
For n training examples, we average the cross entropy loss over the n examples.

![image](https://user-images.githubusercontent.com/59906096/146250927-218b678e-945e-4808-afcb-bb6d2ccd8584.png)

The starter code already converts labels into one hot representations.
Instead of batch gradient descent or stochastic gradient descent, the common practice is to use mini batch gradient descent for deep learning tasks. In this case, the cost function is defined as follows:

![image](https://user-images.githubusercontent.com/59906096/146250970-96ef4edf-95c6-436b-b766-b7887bfe7476.png)

where B is the batch size, i.e. the number of training example in each mini-batch.

1. Unregularized Model: 
Both the forward-propagation and back-propagation for the above loss function are implemented here. The weights of the network are initialized by sampling values from a standard normal distribution.  The bias/intercept term are initialized to 0. The number of hidden units are set to 300, learning rate as 5, and B as 1,000 (mini batch size). This means that 1,000 examples are trained in each iteration.
Therefore, for each epoch, 50 iterations are needed to cover the entire training data. The images are
pre-shuffled, so only sequentially creating mini-batches is good enough. After training the model, the
training for 30 epochs was ran. At the end of each epoch, the value of loss function averaged
over the entire training set was calculated, and plotted (y-axis) against the number of epochs (x-axis). In the
same image, the value of the loss function averaged over the dev set was plotted against the
number of epochs. Similarly, in a new image, the accuracy (on y-axis) over the training set,
measured as the fraction of correctly classified examples, versus the number of epochs (x-axis) was also plotted, along with the
accuracy over the dev set versus number of epochs.

![image](https://user-images.githubusercontent.com/59906096/146252837-fc7b6653-d81e-42e2-9c00-6593f26527ed.png)


2. Regularized Model: 
Added a regularization term to the cross entropy loss. The loss function will become:
![image](https://user-images.githubusercontent.com/59906096/146251253-9e7eac44-2d42-4626-86ed-2eb77740724c.png)

![image](https://user-images.githubusercontent.com/59906096/146252947-7c8923f0-af32-4d35-ba75-ed5947110845.png)


3. Final Test: 
Measuring the model performance on the test set is done here. 
Initialized model from the parameters saved in the non-regularized model, and
evaluated the model performance on the test data. Repeated this using the parameters saved in the regularized model.

![image](https://user-images.githubusercontent.com/59906096/146253037-be3a25b3-5e61-440d-94b1-48b1963a0296.png)


