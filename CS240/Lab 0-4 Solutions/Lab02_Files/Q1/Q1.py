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

    # Student code start TASK 2 : Write l2-loss in form of X, Y, w matrices
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

    # Student code start TASK 3 : Write l2-loss-derivative in form of X, Y, w matrices
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

    # Student code start TASK 4 : Write the code for gradient descent as described above


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

    d = X_train.shape[1]

    ### UNCOMMENT & RUN THE CODE BELOW AFTER COMPLETING  function w_closed_form(X, Y)###
    '''
    w_closed = torch.from_numpy(w_closed_form(X_train,Y_train)).unsqueeze(1)    # closed form solution for w
    '''
    #####   

    # ### UNCOMMENT & RUN THE CODE BELOW AFTER COMPLETING  function l2_loss(X, Y, w)
    '''
    l2_loss_train = l2_loss(X_train,Y_train, w_closed)
    l2_loss_train = np.array([l2_loss_train])
    np.savetxt('l2_loss.txt', l2_loss_train, fmt="%f")
    '''
    #####

    
    ### UNCOMMENT & RUN THE CODE BELOW AFTER COMPLETING  functions l2_loss_derivative(X, Y, w) and train_model(X_train, Y_train, w, eta)
    '''
    w = torch.randn(d,1)
    eta = args.eta
    w_trained, test_err = train_model(X_train, Y_train, X_test, Y_test, w, eta)
    l2_loss_grad = np.array([l2_loss(X_train, Y_train, w_trained)])
    w_trained = w_trained.detach().numpy().squeeze(axis=1)
    np.savetxt('w_trained.txt', w_trained, fmt="%f")
    test_err = np.array(test_err)
    np.savetxt('test_err.txt', test_err, fmt="%f")
    '''
    # #####


    # TASK 5 : Plot the early stopping criterion
    ### UNCOMMENT & RUN THE CODE BELOW AFTER TRAINING THE ABOVE MODEL

    
    # Code to find e_star: the epoch after which test error starts increasing
    '''
    for e_star in range(len(test_err)-1):
        if test_err[e_star+1] > test_err[e_star]:
            break;

    #Code for plotting and saving the figure
    xvals = np.arange(0,len(test_err))
    plt.plot(xvals[e_star-500:e_star+500], test_err[e_star-500:e_star+500])
    plt.xlabel("iterations")
    plt.ylabel("test error")
    plt.show()
    '''

    #####





