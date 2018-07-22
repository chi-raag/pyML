class stochasticGD(object):
    """ADAptive LInear NEuron classifier

    Parameters
    --------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset
    shuffle : bool (default : true)
        Shuffles training data every epoch if True to prevent cycles
    random_state : int
        Random number generator seed for random weight initialization

    Attributes
    ---------------
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        Sum-of-squares cost function value averaged over all training samples in each epoch
    """

    def __init__(self, eta=.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ---------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors
        y : array-like, shape = [n_samples]
            Target values

        Returns
        ---------------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit traiing data without reinitializing the weights"""
        if not self.w_initialized:

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
