import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_contour_diagram(X, y):
    '''
    @params
        X : 2D numpy array of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        y : 1D numpy array of size(n,1)
    '''
    n,d = X.shape
    dim = 200
    a = np.linspace(0, 8, dim).reshape(-1,1)
    b = np.linspace(0, 8, dim).reshape(-1,1)

    A, B = np.meshgrid(a, b)

    MSE = np.sum(np.square((y.reshape(1,1,-1) - (a @ X[:,0].reshape(1,-1)).reshape(1,dim,-1) - (b @ X[:,1].reshape(1,-1)).reshape(dim,1,-1))), axis=2)/n

    fig, ax = plt.subplots()
    levels = np.square(np.arange(0, 4, 0.25))
    CS = ax.contour(A, B, MSE, levels)
    ax.clabel(CS, fontsize=9, inline=True)


def gradient_descent(X, y, eta, max_steps):
    '''
    @params
        X : 2D numpy array of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        y : 1D numpy array of size(n,1)
        eta : learning rate
        max_steps : no of steps to run gradient descent
    @returns
        w_values : 2D numpy array of size(max_steps+1, d)
    '''
    n,d = X.shape
    w = np.zeros((d,1))

    w_values = []
    w_values.append(w)

    ### TASK 1 : Write gradient descent code here ###

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

    w_values = np.array(w_values).reshape(-1,d)

    assert w_values.shape == (max_steps+1, d)
    return w_values


def stochastic_gradient_descent(X, y, eta, max_steps):
    '''
    @params
        X : 2D numpy array of size(n,d)
        n : no of samples for the X dataset
        d : dimension of each sample vector x(i)
        y : 1D numpy array of size(n,1)
        eta : learning rate
        max_steps : no of steps to run gradient descent
    @returns
        w_values : 2D numpy array of size(max_steps+1, d)
    '''
    n,d = X.shape
    w = np.zeros((d,1))

    w_values = []
    w_values.append(w)

    ### TASK 1 : Write stochastic gradient descent code here ###

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

    w_values = np.array(w_values).reshape(-1,d)

    assert w_values.shape == (max_steps+1, d)
    return w_values.reshape(-1,d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="The name of the dataset to be used" )
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)

    X = data[['x_1', 'x_2']].to_numpy()
    y = data['y'].to_numpy().reshape(-1,1)


    w_values = gradient_descent(X, y, args.eta, args.steps)
    np.savetxt('w_values_GD.csv', w_values, delimiter=',')

    plot_contour_diagram(X, y)
    plt.scatter(w_values[:,0], w_values[:,1], c='r', marker='x', label='GD')
    plt.ylabel('w_2')
    plt.xlabel('w_1')
    plt.title('Gradient Descent')
    plt.savefig('contour_plot_GD.png')
    

    w_values = stochastic_gradient_descent(X, y, args.eta, args.steps)
    np.savetxt('w_values_SGD.csv', w_values, delimiter=',')

    plot_contour_diagram(X, y)
    plt.scatter(w_values[:,0], w_values[:,1], c='b', marker='x', label='SGD')
    plt.ylabel('w_2')
    plt.xlabel('w_1')
    plt.title('Stochastic Gradient Descent')
    plt.savefig('contour_plot_SGD.png')