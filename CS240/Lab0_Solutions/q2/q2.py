import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA

    standard = init_array - init_array.mean()
    covariance = np.cov(standard.T, bias = 0)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = np.take_along_axis(eigenvalues, sorted_indices, axis = 0)
    sorted_eigenvalues = np.round(sorted_eigenvalues, 4)

    top_eigenvectors = (eigenvectors.T[sorted_indices[: dimensions]]).T
    final_data = np.matmul(np.array(standard), top_eigenvectors)

    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png

    plt.scatter(final_data[:,0], final_data[:,1])
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.savefig("out.png")

    # END TODO
