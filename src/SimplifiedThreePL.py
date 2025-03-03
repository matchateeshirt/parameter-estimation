import numpy as np
import scipy.optimize as opt
from src.Experiment import Experiment

class SimplifiedThreePL:
    def __init__(self, experiment: Experiment):
        """
        Initializes the SimplifiedThreePL model with an Experiment object.
        """
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an instance of Experiment class")
        
        self.experiment = experiment
        self._discrimination = None  # Discrimination parameter (a)
        self._logit_base_rate = None  # Logit of base rate parameter (c)
        self._is_fitted = False
    
    def summary(self):
        """
        Returns a summary dictionary containing total trials, correct trials,
        incorrect trials, and number of conditions.
        """
        return {
            "n_total": sum(self.experiment.n_correct) + sum(self.experiment.n_incorrect),
            "n_correct": sum(self.experiment.n_correct),
            "n_incorrect": sum(self.experiment.n_incorrect),
            "n_conditions": len(self.experiment.n_correct),
        }
    
    def predict(self, parameters):
        """
        Returns probability of correct response given parameters.
        """
        a, logit_c = parameters
        c = 1 / (1 + np.exp(-logit_c))  # Convert logit_c back to c
        difficulties = np.array([2, 1, 0, -1, -2])  # Fixed difficulty values
        probabilities = c + (1 - c) / (1 + np.exp(-a * (-difficulties)))
        return probabilities
    
    def negative_log_likelihood(self, parameters):
        """
        Computes the negative log-likelihood of the data given parameters.
        """
        probabilities = self.predict(parameters)
        log_likelihood = np.sum(
            self.experiment.n_correct * np.log(probabilities)
            + self.experiment.n_incorrect * np.log(1 - probabilities)
        )
        return -log_likelihood
    
    def fit(self):
        """
        Uses maximum likelihood estimation to fit the model.
        """
        initial_guess = [1.0, 0.0]  # Initial values for a and logit_c
        result = opt.minimize(self.negative_log_likelihood, initial_guess, method='Nelder-Mead')
        
        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._is_fitted = True
        else:
            raise RuntimeError("Optimization failed")
    
    def get_discrimination(self):
        """
        Returns the estimated discrimination parameter.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet")
        return self._discrimination
    
    def get_base_rate(self):
        """
        Returns the estimated base rate parameter (not logit_c!).
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet")
        return 1 / (1 + np.exp(-self._logit_base_rate))  # Convert logit_c to c
    
