import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

class NNEpidemicPrediction:
    def __init__(self, X, y, country="Dataset"):
        self.X          = np.array(X, dtype=float)
        self.y          = np.array(y, dtype=float)
        self.country    = country
        self.scaler     = StandardScaler()
        self.model      = None
        self.rms        = None   
        self.best_alpha = None    
    
    def scalePredictors(self):
        """ Scales the predictors using StandardScaler. """
        self.X = self.scaler.fit_transform(self.X)

    def splitData(self, test_size=0.8):
        split = int(test_size * len(self.X))

        return self.X[:split], self.X[split:], self.y[:split], self.y[split:]
    
    def trainNN(self, hidden=(64,32), max_iter=1000, alpha=1e-4):
        """ Trains the neural network with given hyperparameters. """
        X_train, X_test, y_train, y_test = self.splitData()
        self.scaler.fit(X_train)
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model = MLPRegressor(hidden_layer_sizes=hidden, 
                                  activation='relu',
                                  solver='adam',
                                  alpha=alpha, max_iter=max_iter,
                                  random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        self.rms = mean_squared_error(y_test, y_pred)
        print(f"{self.country} NN RMS Error: {self.rms:.3f}")

    def find_best_alpha(self, alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
        """ Uses GridSearchCV to find the best alpha """
        self.scalePredictors()
        param_grid = {'alpha': alphas}

        grid = GridSearchCV(MLPRegressor(hidden_layer_sizes=(64,32),
                                        activation='relu',
                                        solver='adam',
                                        max_iter=1000,
                                        random_state=42),
                            param_grid,
                            scoring='neg_mean_squared_error',
                            cv=5)
        grid.fit(self.X, self.y)
        self.best_alpha = grid.best_params_['alpha']
        print(f"{self.country} Best fit alpha: {self.best_alpha}")

    def predict(self, X_new):
        """ Predicts using the trained model on new data. """
        
        X_new = self.scaler.transform(np.array(X_new, dtype=float))
        return self.model.predict(X_new)
    
    def plot_predictions(self, time=None):
        y_pred = self.model.predict(self.scaler.transform(self.X))
        plt.figure()
        if time is not None:
            plt.plot(time, self.y, label='Actual', marker='o')
            plt.plot(time, y_pred, label='Predicted', marker='x')
            plt.xlabel('Time [days]')
        else:
            plt.plot(self.y, label='Actual', marker='o')
            plt.plot(y_pred, label='Predicted', marker='x')
            plt.xlabel('Samples')
        plt.title(f'Neural Network Predictions for {self.country}')
        plt.ylabel('Infected Individuals')
        plt.legend()
        plt.grid()
        plt.show()
