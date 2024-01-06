import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA

    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png

    # END TODO
