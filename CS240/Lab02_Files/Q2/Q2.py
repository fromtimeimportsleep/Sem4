import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def train_test_split(dataframe):
    return dataframe.iloc[0:240], dataframe.iloc[240:300]


def w_closed_form(X, Y):
    '''
    @params
        X : 2D tensor of shape(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of shape(n,1)
    calculates w_closed : 1D tensor of shape(d,1)
    writes the w_closed as a numpy array into the text file "w_closed.txt"
    returns w_closed
    '''

    # Student code start TASK 1 : Write w_closed in form of X, Y matrices
    # Round w_closed upto 4 decimal places
    # w_closed = YOUR CODE HERE

    
    # Student code end

    w_closed = w_closed.detach().numpy().squeeze(axis=1)
    np.savetxt('w_closed.txt', w_closed, fmt="%f")
    return w_closed

def transform_features(X, degree=1):
    '''
    For Q3
    Args:
    - X: Array containing the feature vectors.
    - degree : The degree of the polynomial to which the features are to be transformed
    
    Returns:
    - phi_X : Array containing the feature vectors with the transformed features concatenated
    '''
    #Implement the polynomial basis function transformation, and return it
    phi_X = None
    
    # Student code start TASK 1 : Write the code for polynomial basis function transformation
    # phi_X = YOUR CODE HERE

    # Student code end
    
    return phi_X


def l2_loss(X, Y, w):
    '''
    @params
        X : 2D tensor of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of size(n,1)
        w : 1D tensor of size(d,1)
    return loss : np.float64 : scalar real value
    '''

    w = w.double()

    # Student code start TASK 3 : Write l2-loss in form of X, Y, w matrices
    # Please take care of normalization factor 1/n
    # loss = YOUR CODE HERE

    # Student code end

    return (loss)


def l2_loss_derivative(X, Y, w):
    '''
    @params
        X : 2D tensor of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of size(n,1)
        w : 1D tensor of size(d,1)
    return derivative : 1D tensor of size(d,1)
    '''

    w = w.double()

    # Student code start TASK 4 : Write l2-loss-derivative in form of X, Y, w matrices
    # Please take care of normalization factor 1/n

    # derivative = YOUR CODE HERE

    # Student code end

    return (derivative)


def train_model(X_train, Y_train, X_test, Y_test, w, eta):
    '''
    @params
        X_train : 2D tensor of size(n,d) over which model is trained
        n : no of samples for the X_train dataset
        d : dimension of each sample vector x(i)
        Y_train : 1D tensor of size(n,1) over which model is trained
        w : initial weights vector (that needs to be optimised using gradient descent)
        eta : learning rate
    @returns
        w : 1D tensor of size(d,1) ,  the final optimised w
        iters : Total iterations it take for algorithm to converge
        test_err : python list containing the l2-loss at each iteration

    '''

    epsilon = 1e-15  # Stopping precision
    old_loss = 0
    test_err = []  # Initially empty list

    '''
    stopping condition: abs(new_loss - old_loss) <= epsilon

    Pseudo code:

    while stopping condition not met:    
        calculate old loss
        calculate gradient (dw)
        update w = w - eta*dw
        append test error to test_err (l2_loss)
    
    '''

    # Student code start TASK 5 : Write the code for gradient descent as described above


     # Student code end

    return w, test_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The name of the dataset to be used" )
    parser.add_argument('--seed', type = int, default = 335)
    parser.add_argument('--eta', type=float, default=1e-3)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    data = pd.read_csv(args.dataset , index_col=0)

    data_train, data_test = train_test_split(data)

    X_train = (data_train.iloc[:,:-1].to_numpy())
    Y_train = (data_train.iloc[:,-1].to_numpy())
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train).unsqueeze(1)

    X_test = (data_test.iloc[:,:-1].to_numpy())
    Y_test = (data_test.iloc[:,-1].to_numpy())
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test).unsqueeze(1)

    possible_degrees = range(1, 11)


    # UNCOMMENT & RUN THE CODE BELOW AFTER COMPLETING  function transform_features(X, degree=1)
    '''
    for degree in possible_degrees:
        transformed_X_train = transform_features(X_train, degree)
        transformed_X_test = transform_features(X_test, degree)
        d = transformed_X_train.shape[1]
        w_closed = torch.from_numpy(w_closed_form(transformed_X_train,Y_train)).unsqueeze(1)    # closed form solution for w
        l2_loss_train = float(l2_loss(transformed_X_train,Y_train, w_closed))
        l2_test_loss = float(l2_loss(transformed_X_test,Y_test,w_closed))
        print(degree, l2_loss_train, l2_test_loss)

    optimal_degree = 0

    print("optimal degree", optimal_degree)
    '''





