import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

class EbolaDataLoader:
    def __init__(self, files: dict):
        """Initialize with a dictionary of country names to file paths."""
        self.files = files
        self.data = {}  

        # epidemiological parameters
        self.gamma = 1.0 / 7.0  # infectious period ~7 days
        self.sigma = 1.0 / 9.7  # incubation period ~9.7 days
        self.N     = 1e7        # total population

    # Data handling methods
    def load_country_data(self, country: str):
        path = self.files[country]
        new_cases = np.loadtxt(path, skiprows=1, usecols=[2])
        new_cases = new_cases.astype(float)
        days = np.arange(len(new_cases), dtype=float)
        cum_cases = np.cumsum(new_cases)
        self.data[country] = (days, new_cases, cum_cases)
        return days, new_cases, cum_cases

    def get_data(self, country: str):
        """Return cached data if available, otherwise load fresh."""
        if country not in self.data:
            return self.load_country_data(country)
        return self.data[country]
    
    # SEZR model methods
    def f_SEZR(self, y, t, b0, lam):
        """SEZR model differential equations."""
        S, E, Z, R = y
        beta_t = b0 * math.exp(-lam * t) 
        dSdt = -beta_t * (S * Z) / self.N
        dEdt = beta_t * (S * Z) / self.N - self.sigma * E
        dZdt = self.sigma * E - self.gamma * Z
        dRdt = self.gamma * Z
        return np.array([dSdt, dEdt, dZdt, dRdt], float)
    
    # Numerical solver using Runge-Kutta 4th order method
    def rk4(self, f, y0, t0, t1, dt):
        n = int(np.ceil((t1 - t0) / dt))
        t = np.linspace(t0, t1, n + 1)
        y = np.zeros((n + 1, len(y0)), float)
        y[0] = y0
        for i in range(n):
            h   = t[i+1] - t[i]
            yi  = y[i]
            ti  = t[i]

            k1 = f(yi, ti)
            k2 = f(yi + 0.5 * h * k1, ti + 0.5 * h)
            k3 = f(yi + 0.5 * h * k2, ti + 0.5 * h)
            k4 = f(yi + h * k3, ti + h)
            y[i+1] = yi + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return t, y
    
    def simulate_SEZR(self, beta0, lam, t_days, Z0=1.0):
        y0 = np.array([self.N - Z0, 0.0, Z0, 0.0], float)  # Initial conditions: S, E, Z, R
        t, Y = self.rk4(lambda y, t: self.f_SEZR(y, t, beta0, lam), y0, 0.0, t_days[-1], 1.0)
        S, E, Z, R = Y.T
        return t, R
    
    def non_linear_SEZR(self, t, beta0, lam,days):
        t_model, R = self.simulate_SEZR(beta0, lam, days)
        return np.interp(t, t_model, R)

    # Parameter fitting
    def fit_parameters(self, country: str, beta0_guess=0.5, lam_guess=0.01):
        days, new_cases, cum_cases = self.get_data(country)

        def model_func(t, beta0, lam):
            return self.non_linear_SEZR(t, beta0, lam, days)
        
        bounds = ([0.0, 0.0], (np.inf, np.inf))

        popt, pcov = curve_fit(model_func, days, cum_cases, p0=[beta0_guess, lam_guess], bounds=bounds)
        beta0_fit, lam_fit = popt

        return beta0_fit, lam_fit

    # Plotting
    def plot_model_vs_data(self, country: str, beta0: float, lam: float):
        days, new_cases, cum_cases = self.get_data(country)
        t_model, R = self.simulate_SEZR(beta0, lam, days)

        plt.figure()
        plt.plot(days, cum_cases, label="Data (cumulative)")
        plt.plot(days, np.interp(days, t_model, R), label="Model R(t)")
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative cases")
        plt.title(f"{country}: Model vs Data (β0={beta0:.2f}, λ={lam:.2f})")
        plt.legend()
        plt.show()
