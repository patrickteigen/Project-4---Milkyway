import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegressionFit:
    def __init__(self, x, y, country="Dataset"):
        self.x          = np.array(x, dtype=float)
        self.y          = np.array(y, dtype=float)
        self.country    = country
        self.coeffs     = None  
        self.degree     = None  

    def polyFit (self, m): 
        """ 
        Fit a polynomial of degree m to data points (x, y) using normal equations.
        Returns the coefficients [a0, a1, a2, ..., am] of the fitted polynomial.
        """

        n = len(self.x)

        # Build matrix A and vector b for normal equations
        A = np.zeros((m+1,m+1))
        b = np.zeros(m+1)

        for k in range(m+1):
            for j in range(m+1):
                A[k,j] = np.sum(self.x**(k+j))
            b[k] = np.sum(self.y * (self.x**k))

        # Solve for coefficients
        self.coeffs = np.linalg.solve(A, b)
        self.degree = m
        return self.coeffs

    def polyPredict(self, x_new):
        """
        Predict y values for given x_new using polynomial coefficients.
        Coeffs should be in the form [a0, a1, a2, ..., am].
        """
        x_new = np.array(x_new, dtype=float)
        y_pred = np.zeros_like(x_new, dtype=float)

        for j, a in enumerate(self.coeffs):
            y_pred += a * (x_new ** j)
        return y_pred
    
    def fit_quality(self):
        """ 
        Compute standard deviation of residuals.
        """
        y_fit = self.polyPredict(self.x)
        residuals = self.y - y_fit
        S = np.sum(residuals**2)
        n, m = len(self.x), self.degree
        sigma = np.sqrt(S/(n-m))
        return sigma
    
    def plot_fit(self):
        """
        Plot data point and fitted polynomial curve. 
        """

        plt.figure()
        plt.scatter(self.x, self.y, label='Data Points', color='blue')
        plt.plot(self.x, self.polyPredict(self.x), label=f"Fitted Polynomial (degree {self.degree})", color='red')
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative Cases")
        plt.title(f"Polynomial Regression Fit for {self.country}")
        plt.legend()
        plt.grid()
        plt.show()