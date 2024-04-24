from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    
    return v * (t - (1 - np.exp(-k * t)) / k)

    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k

    popt, _ = scipy.optimize.curve_fit(func, df["t"], df["S"])
    v, k = popt

    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png

    plt.plot(df["t"], df["S"], 'b*', label='data')
    plt.plot(df["t"], func(df["t"], v, k), 'r-', label=f'fit: v={v}, k={k}')

    plt.xlabel('t')
    plt.ylabel('S')
    plt.legend()
    plt.savefig("fit_curve.png")

    # END TODO
