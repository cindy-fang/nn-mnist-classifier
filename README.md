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

The starter code already converts labels into one hot representations for you.
Instead of batch gradient descent or stochastic gradient descent, the common practice is to use mini batch gradient descent for deep learning tasks. In this case, the cost function is defined as follows:

![image](https://user-images.githubusercontent.com/59906096/146250970-96ef4edf-95c6-436b-b766-b7887bfe7476.png)

where B is the batch size, i.e. the number of training example in each mini-batch.

1. Unregularized Model: 
Implement both forward-propagation and back-propagation for the above loss function. Initialize
the weights of the network by sampling values from a standard normal distribution. Initialize the
bias/intercept term to 0. Set the number of hidden units to be 300, and learning rate to be 5.
Set B = 1,000 (mini batch size). This means that we train with 1,000 examples in each iteration.
Therefore, for each epoch, we need 50 iterations to cover the entire training data. The images are
pre-shuffled. So you don’t need to randomly sample the data, and can just create mini-batches
sequentially. Train the model with mini-batch gradient descent as described above. Run the
training for 30 epochs. At the end of each epoch, calculate the value of loss function averaged
over the entire training set, and plot it (y-axis) against the number of epochs (x-axis). In the
same image, plot the value of the loss function averaged over the dev set, and plot it against the
number of epochs. Similarly, in a new image, plot the accuracy (on y-axis) over the training set,
measured as the fraction of correctly classified examples, versus the number of epochs (x-axis).
In the same image, also plot the accuracy over the dev set versus number of epochs.

2. Regularized Model: 
Now add a regularization term to your cross entropy loss. The loss function will become:
![image](https://user-images.githubusercontent.com/59906096/146251253-9e7eac44-2d42-4626-86ed-2eb77740724c.png)

3. Final Test: 
All this while you should have stayed away from the test data completely. Now that you have
convinced yourself that the model is working as expected (i.e, the observations you made in the
previous part matches what you learnt in class about regularization), it is finally time to measure
the model performance on the test set. Once we measure the test set performance, we report it
(whatever value it may be), and NOT go back and refine the model any further.
Initialize your model from the parameters saved in part (a) (i.e, the non-regularized model), and
evaluate the model performance on the test data. Repeat this using the parameters saved in part
(b) (i.e, the regularized model).
Report your test accuracy for both regularized model and non-regularized model. You should
have accuracy close to 0.9318 without regularization, and with 0.9670 regularization. Briefly (in
one sentence) explain why this outcome makes sense.
