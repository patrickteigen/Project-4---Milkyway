import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionFit:
    # Initialize with data points and optional country name
    def __init__(self, x, y, country="Dataset"): 
        self.x          = np.array(x, dtype=float)
        self.y          = np.array(y, dtype=float)
        self.country    = country
        self.a          = None # slope
        self.b          = None # intercept

    # Perform linear regression to find slope and intercept
    def fit(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)

        self.b = np.sum((self.x - x_mean) * (self.y - y_mean)) / np.sum((self.x - x_mean) ** 2)
        self.a = y_mean - self.b * x_mean
        return self.a, self.b
    
    # Predict y values for new x values using the fitted line
    def predict(self, x_new):
        if self.a is None or self.b is None:
            raise ValueError("Model is not fitted yet. Call the fit() method first.")
        return self.a + self.b * np.array(x_new)
    
    # Plot the original data points and the fitted line
    def plot(self):
        plt.figure()
        plt.scatter(self.x, self.y, color='blue', label='Data Points')
        plt.plot(self.x, self.predict(self.x), label=f"Fitted Line: y = {self.a:.2f} + {self.b:.2f}x", color='red')
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative Cases")
        plt.title(f"Linear Regression Fit for {self.country}")
        plt.legend()
        plt.grid()
        plt.show()