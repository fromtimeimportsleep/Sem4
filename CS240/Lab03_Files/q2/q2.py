import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def train_test_split(dataframe):
    total_samples = dataframe.shape[0]
    train_ratio = .8
    random_indices = np.random.permutation(total_samples)
    train_set_size = int(train_ratio * total_samples)
    train_indices = random_indices[:train_set_size]
    test_indices = random_indices[train_set_size:]
    return dataframe.iloc[train_indices], dataframe.iloc[test_indices]


def lasso_loss(X, Y, w, lambd):
    '''
    @params
        X : 2D tensor of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of size(n,1)
        w : 1D tensor of size(d,1)
        lambd : regularization parameter
    return loss : np.float64 : scalar real value
    '''

    w = w.double()
    loss = None

    # Student code start TASK 1 : Write lasso-loss in form of X, Y, w matrices
    # Please take care of normalization factor 1/n

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

    return (loss)


def lasso_loss_derivative(X, Y, w, lambd):
    '''
    @params
        X : 2D tensor of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        Y : 1D tensor of size(n,1)
        w : 1D tensor of size(d,1)
        lambd : regularization parameter
    return derivative : 1D tensor of size(d,1)
    '''

    w = w.double()
    derivative = None

    # Student code start TASK 2 : Write lasso-loss-derivative in form of X, Y, w matrices
    # Please take care of normalization factor 1/n

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

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

    epsilon = 1e-8  # Stopping precision
    lambd = 1e-3  # Regularization parameter
    old_loss = 0
    test_err = []  # Initially empty list

    np.savetxt('lasso_loss_init.txt', lasso_loss(X_train, Y_train, w, lambd).detach().numpy().reshape(1,1), fmt="%f")

    while (abs(lasso_loss(X_train, Y_train, w, lambd) - old_loss) > epsilon):
        old_loss = lasso_loss(X_train, Y_train, w, lambd)  # compute loss
        dw = lasso_loss_derivative(X_train, Y_train, w, lambd)  # compute derivate
        w = w - eta * dw  # move in the opposite direction of the derivate
        test_err.append(lasso_loss(X_test, Y_test, w, lambd))

    return w, test_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="The name of the dataset to be used" )
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

    d = X_train.shape[1]

    w = torch.randn(d,1)
    eta = 1e-3
    w_trained, test_err = train_model(X_train, Y_train, X_test, Y_test, w, eta)
    w_trained = w_trained.detach().numpy().squeeze(axis=1)
    np.savetxt('w_trained.txt', w_trained, fmt="%f")
    test_err = np.array(test_err)
    np.savetxt('test_err.txt', test_err, fmt="%f")
