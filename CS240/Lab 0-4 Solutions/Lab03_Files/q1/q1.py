import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def data_generate(dataset_filename):
    '''
    Generates the data, returns X and Y in the appropriate shape.
    Return: A dictionary containing X_train, Y_train, X_test and Y_test
    '''
    data = pd.read_csv(dataset_filename)

    total_samples = data.shape[0]
    train_ratio = 0.8
    random_indices = np.random.permutation(total_samples)
    train_set_size = int(train_ratio * total_samples)

    train_indices =  random_indices[:train_set_size]
    test_indices = random_indices[train_set_size:]

    data.iloc[train_indices], data.iloc[test_indices] 
    X_train = (data.iloc[train_indices].iloc[:,:-1]).to_numpy()     # Design matrix for train data 
    y_train = (data.iloc[train_indices].iloc[:,-1]).to_numpy()      # Labels for train data
    y_train = y_train.reshape((y_train.shape[0],1))

    X_test = (data.iloc[test_indices].iloc[:,:-1]).to_numpy()       # Design matrix for test data
    y_test = (data.iloc[test_indices].iloc[:,-1]).to_numpy()        # Labels for test data
    y_test = y_test.reshape((y_test.shape[0],1))

    return {'X_train': X_train, 'Y_train':y_train,
            'X_test': X_test, 'Y_test': y_test}


def create_weights(data_dictionary, lambda_val):
    '''
    Creates the weights matrix using the closed form solution of ridge regression
    Input:
        data_dictionary: A dictionary containing X_train, Y_train, X_test and Y_test
        lambda_val: The hyperparameter value (of lambda) for the ridge regression
    Output: The weights matrix
    '''

    weights = None

    # TODO : Add the code to find the weights vector for a given lambda value
    
    pass
    
    # END TODO
    
    return weights

    
def generate_test_error(data_dictionary, weights):
    '''
    Generates the test error value for a particular weights matrix
    Input:
        data_dictionary: A dictionary containing X_train, Y_train, X_test and Y_test
        weights: The weights matrix generated through create_weights function
    Output: 
        The test error [float]
    '''

    test_error = None

    # TODO : Add the code to find the test error value for a given weights vector
    
    pass
    
    # END TODO
    
    return test_error


def find_optimal_lambda(error_array, lambda_array):
    '''
    Return the optimal value of hyperparameter lambda, which minimizes the test error
    Input:
        error_array: Array of the found test errors
        lambda_array: Array of the lambda values used
    '''

    optimal_lambda = None

    # TODO : Find the best value of hyperparameter lambda

    pass
    
    # END TODO

    return optimal_lambda


def plot_errors(error_array, lambda_array):
    '''
    Plots test error vs lambda and saves it to 'test_errors.png' (UNGRADED)
    Input:
        error_array: Array of the found test errors
        lambda_array: Array of the lambda values used
    '''

    # TODO : Add the code to plot the errors vs lambda, and save it to 'test_errors.png'

    pass 
    
    # END TODO


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filename', type=str, default='dataset.csv', 
                        help="Name of the dataset file to be used" )
    parser.add_argument('--weights_file', type=str, default='weights.csv', 
                        help="Name of the file where weights will be saved")
    parser.add_argument('--error_file', type=str, default='errors.csv', 
                        help="Name of the file where errors will be saved")
    parser.add_argument('--hidden', action='store_true', help="Evaluate on hidden test set")
    args = parser.parse_args()
    
    if args.hidden:
        args.dataset_filename = "hidden_" + args.dataset_filename
        args.weights_file = "hidden_" + args.weights_file
        args.error_file = "hidden_" + args.error_file
    
    np.random.seed(20)
    data_dictionary = data_generate(args.dataset_filename)
    lambda_array = []
    weights_array = []
    error_array = []
    
    # Perform regression for different values of hyperparamter lambda

    for lam in np.arange(0, 2, 0.04):
        lambda_array.append(np.round(lam, 4))
        w = create_weights(data_dictionary, lam)
        error = generate_test_error(data_dictionary, w)
        weights_array.append(w)
        error_array.append(error)
    
    # Print optimal value of hyperparameter lambda
        
    error_array = np.array(error_array)
    weights_array = np.squeeze(np.array(weights_array), axis=-1)
    optimal_lambda = find_optimal_lambda(error_array, lambda_array)
    print(optimal_lambda)

    # Save generated weights and errors

    df_lambda = pd.DataFrame(lambda_array, columns=["lambda"])

    column_names = [f"coeff_feature_{i}" for i in range(1, len(weights_array[0]) + 1)]
    df_weights = pd.DataFrame(weights_array, columns=column_names)

    df_weights = pd.concat([df_lambda, df_weights], axis=1)
    df_weights.to_csv(args.weights_file, index=False)

    df_error = pd.DataFrame(error_array, columns=["error"])
    df_error = pd.concat([df_lambda, df_error], axis=1)
    df_error.to_csv(args.error_file, index=False)

    if not args.hidden:
        plot_errors(error_array, lambda_array)
    