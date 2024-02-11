import numpy as np 
import pandas as pd 	


def pre_processing(df):

	"""
    Partition the data into features and target.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    tuple: Features (X) and target (y).
    """


	X = df.drop([df.columns[-1]], axis = 1)
	y = df[df.columns[-1]]

	return X, y

def train_test_split(x, y, test_size = 0.25, random_state = None):

	"""
    Partition the data into train and test sets.

    Parameters:
    - x (array-like): Features.
    - y (array-like): Target.
    - test_size (float, optional): Percentage of data to be used for testing. Default is 0.25.
    - random_state (int or None, optional): Seed for random number generation. Default is None.

    Returns:
    tuple: x_train, x_test, y_train, y_test.
    """


	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test




class  NaiveBayes:

	"""
		Bayes Theorem:
										Likelihood * Class prior probability
				Posterior Probability = -------------------------------------
											Predictor prior probability
				
							  			 P(x|c) * p(c)
							   P(c|x) = ------------------ 
											  P(x)
	"""

	def __init__(self):

		"""
        Initialize NaiveBayes attributes.
        """

		"""
			Attributes:
				likelihoods: Likelihood of each feature per class
				class_priors: Prior probabilities of classes 
				pred_priors: Prior probabilities of features 
				features: All features of dataset

		"""
		self.features = list
		self.likelihoods = {}
		self.class_priors = {}
		self.pred_priors = {}

		self.X_train = np.array
		self.y_train = np.array
		self.train_size = int
		self.num_feats = int

	def fit(self, X, y):

		"""
        Fit the Naive Bayes model to the training data.

        Parameters:
        - X (array-like): Training features.
        - y (array-like): Training target.
        """

		self.features = list(X.columns)
		self.X_train = X
		self.y_train = y
		self.train_size = X.shape[0]
		self.num_feats = X.shape[1]

		for feature in self.features:
			self.likelihoods[feature] = {}
			self.pred_priors[feature] = {}

			for feat_val in np.unique(self.X_train[feature]):
				self.pred_priors[feature].update({feat_val: 0})

				for outcome in np.unique(self.y_train):
					self.likelihoods[feature].update({feat_val+'_'+outcome:0})
					self.class_priors.update({outcome: 0})


		self._calc_class_prior()
		self._calc_likelihoods()
		self._calc_predictor_prior()



	def _calc_class_prior(self):


		"""
        Calculate the prior class probability.
        """
		
            # Student code start TASK 1 : Calculate Prior Probability of Classes P(y) from y_train also using train_size

    
		for outcome in np.unique(self.y_train):
			### YOUR CODE BEGINS HERE ###

			#TODO_1

			self.class_priors[outcome] = 0 #TODO_2    Remove 0 and write your code 
    		### YOUR CODE ENDS HERE ###



	def _calc_likelihoods(self):

		"""
        Calculate the likelihood.
        """

		""" P(x|c) - Likelihood """

		for feature in self.features:

			for outcome in np.unique(self.y_train):
				outcome_count = sum(self.y_train == outcome)
				feat_likelihood = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].value_counts().to_dict()

				for feat_val, count in feat_likelihood.items():
					self.likelihoods[feature][feat_val + '_' + outcome] = count/outcome_count


	def _calc_predictor_prior(self):

		"""
        Calculate the predictor prior probability.
        """

		""" P(x) - Evidence """

		for feature in self.features:
			feat_vals = self.X_train[feature].value_counts().to_dict()

			for feat_val, count in feat_vals.items():
				self.pred_priors[feature][feat_val] = count/self.train_size


	def predict(self, X):
		
		"""
        Predict the class labels for the given features.

        Parameters:
        - X (array-like): Features to be predicted.

        Returns:
        array: Predicted class labels.
        """


		""" Calculates Posterior probability P(c|x) """
		
       

		results = []
		X = np.array(X)

         # Student code start TASK 2: Now, Calculate Posterior Probability for each class using the Naive Bayesian equation. The Class with maximum probability is the outcome of the prediction.

	
		for query in X:
			probs_outcome = {}
			for outcome in np.unique(self.y_train):

				prior = None #TODO_1   Take value from task-1 variable  (Remove 'None')
				likelihood = 1
				evidence = 1

				for feat, feat_val in zip(self.features, query):
					likelihood *= self.likelihoods[feat][feat_val + '_' + outcome]
					evidence *= self.pred_priors[feat][feat_val]
				
				posterior = 0 #TODO_2         Remove 0 and get posterior

				probs_outcome[outcome] = posterior

			result = None #TODO_3          Remove None and get the Max
			results.append(result)

		return np.array(results)

			

if __name__ == "__main__":

	#Weather Dataset
	print("Weather Dataset:")

	df = pd.read_table("weather.txt")

	#Split fearures and target
	X,y  = pre_processing(df)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

	nb_clf = NaiveBayes()
	nb_clf.fit(X_train, y_train)

	
	#Query 1:
	query = np.array([['Rainy','Mild', 'Normal', 't']])
	print("Query 1:- {} ---> {}".format(query, nb_clf.predict(query)))

	#Query 2:
	query = np.array([['Overcast','Cool', 'Normal', 't']])
	print("Query 2:- {} ---> {}".format(query, nb_clf.predict(query)))

	#Query 3:
	query = np.array([['Sunny','Hot', 'High', 't']])
	print("Query 3:- {} ---> {}".format(query, nb_clf.predict(query)))
	
	
