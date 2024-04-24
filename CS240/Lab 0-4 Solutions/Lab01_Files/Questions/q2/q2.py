"""
Given labelled training data set (X, Y), we want a learn a linear predictor of the form  f(x) = W^T x, i.e. find out
W which can predict y=f(x) well for unseen data points X_new.

One way to determine W is to minimize the least square error i.e. solve the following
unconstrained optimization problem min (Xw-Y)^{T}(Xw-Y), where w is the variable.

Convert this problem to the standard quadratic programming problem and solve using cvxopt.

Reference for solving quadratic programs:
(1) https://en.wikipedia.org/wiki/Quadratic_programming find relation between least squares minimization and QP in
section (qp problem formulation and constrained least squares)
(2) https://cvxopt.org/examples/tutorial/qp.html - Example
(3) https://cvxopt.org/userguide/coneprog.html#quadratic-programming
"""
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from cvxopt import matrix, solvers
import argparse


def QP_solver(X, Y):
    """ Given labelled data, this function solves the least squared error minimization problem by transforming it into
    a quadratic program.

    :param X: Sample inputs
    :param Y: Sample target outputs
    :return: (numpy array (2,1)) weights w
    """
    ###############################################################
    # %% Student Code Start
    # Implement here
    # %% Student Code End
    ###############################################################
    # return the weights w which is the solution of the QP (as a numpy array (2, 1) dimensions) below
    return None


def parse_commandline_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDirectory', type=str, required=True, help='Directory of the test case files')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    # get command line args
    args = parse_commandline_args()
    if args.testDirectory is None:
        raise ValueError("No file provided")
    # load labelled data
    X = pd.read_csv("{}/X.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    Y = pd.read_csv("{}/Y.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    A = QP_solver(X, Y)
    if A is not None:
        for val in A:
            print(val[0])
        # matrix multiplication
        # plotting graph for visualizing the linear predictor
        y_1 = X @ A
        y_1 = y_1.reshape(len(X))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X[:, 0], X[:, 1], y_1, color="blue", label="least square fit")
        ax.scatter3D(X[:, 0], X[:, 1], Y, s=1.5, label="data points")
        plt.legend()
        plt.savefig('lstsq_QP.png')
