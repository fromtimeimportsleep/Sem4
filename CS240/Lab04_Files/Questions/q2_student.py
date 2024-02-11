import numpy as np
import pandas as pd


class MultiClassLogisticRegression:

    """
    MultiClassLogisticRegression class implements logistic regression for multiple classes.

    Parameters:
    - n_iter (int): Number of iterations for gradient descent. Default is 10000.
    - thres (float): Convergence threshold for gradient descent. Default is 1e-3.

    Methods:
    - fit(X, y, rand_seed=4): Initialize the logistic regression model and prepare the data for training.
    - fit_data(X, y, batch_size=64, lr=0.001, verbose=False): Train the logistic regression model using gradient descent.
    - predict(X): Predict class probabilities for input features X.
    - softmax(z): Apply the softmax function to the input array z.
    - add_bias(X): Add a bias term to the input matrix X.
    - one_hot(y): Convert class labels to one-hot encoding.
    - cross_entropy(y, probs): Compute the cross-entropy loss between true labels y and predicted probabilities probs.
    """
    
    def __init__(self, n_iter = 10000, thres=1e-3):

        """
        Initialize MultiClassLogisticRegression.

        Parameters:
        - n_iter (int): Number of iterations for gradient descent.
        - thres (float): Convergence threshold for gradient descent.
        """

        self.n_iter = n_iter
        self.thres = thres
    
    def fit(self, X, y, rand_seed=4): 

        """
        Initialize the logistic regression model and prepare the data for training.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Class labels.
        - rand_seed (int): Random seed for reproducibility.

        Returns:
        - X (numpy.ndarray): Modified input features.
        - y (numpy.ndarray): One-hot encoded class labels.
        """

        np.random.seed(rand_seed) 
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        return X,y
 
    def fit_data(self, X, y, batch_size=64, lr=0.001, verbose=False):

        """
        Train the logistic regression model using gradient descent.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): One-hot encoded class labels.
        - batch_size (int): Batch size for mini-batch gradient descent.
        - lr (float): Learning rate for gradient descent.
        - verbose (bool): Whether to print training information.

        Returns:
        None
        """

        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.softmax(self.predict(X))))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.softmax(self.predict(X_batch))
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres: break
            i +=1

    
    def predict(self, X):

        """
        Predict class probabilities for input features X.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        numpy.ndarray: Predicted class probabilities.
        """
        

        # Student code start TASK 1 : For each class k compute a linear combination of the input features and the weight vector of class k, that is,for each training example compute a score for each class.

        ### YOUR CODE BEGINS HERE ###
        pre_vals=np.ndarray([0])        #Comment this line and write your code

    #   ## YOUR CODE ENDS HERE ###
        
        return pre_vals
    
    def softmax(self, z):

        """
        Apply the softmax function to the input array z.

        Parameters:
        - z (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Output of the softmax function.
        """


        # Student code start TASK 2 : Write softmax function

        ### YOUR CODE BEGINS HERE ###
        post_softmax=np.ones((64,3))          #Comment this line and write your code

        ### YOUR CODE ENDS HERE ###
        return post_softmax

  
    def add_bias(self,X):

        """
        Add a bias term to the input matrix X.

        Parameters:
        - X (numpy.ndarray): Input matrix.

        Returns:
        numpy.ndarray: Input matrix with an added bias column.
        """

        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):

        """
        Convert class labels to one-hot encoding.

        Parameters:
        - y (numpy.ndarray): Class labels.

        Returns:
        numpy.ndarray: One-hot encoded class labels.
        """

        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    
    def cross_entropy(self, y, probs):

        """
        Compute the cross-entropy loss between true labels y and predicted probabilities probs.

        Parameters:
        - y (numpy.ndarray): True class labels in one-hot encoding.
        - probs (numpy.ndarray): Predicted class probabilities.

        Returns:
        float: Cross-entropy loss.
        """
        post_cross_entropy=0.0
        # Student code start TASK 3 : Write cross entropy loss function

        ### YOUR CODE BEGINS HERE ###

        ### YOUR CODE ENDS HERE ###
        return post_cross_entropy
    
if __name__ == "__main__":

    data = pd.read_csv('iris.csv')
    from sklearn import datasets

    X,y = datasets.load_iris(return_X_y=True)
    lr = MultiClassLogisticRegression(thres=1e-5)
    X,y=lr.fit(X,y)
    lr.fit_data(X,y,lr=0.0001)
    print(lr.weights)
    np.savetxt('output_q2.txt', lr.weights, fmt="%f")