import numpy as np
import matplotlib.pyplot as plt

class EbolaDataLoader:
    def __init__(self, files: dict):
        """
        Initialize with a dictionary of country -> file path.
        Example:
        files = {
            "Guinea": "data/ebola_cases_guinea.dat",
            "Liberia": "data/ebola_cases_liberia.dat",
            "Sierra Leone": "data/ebola_cases_sierra_leone.dat",
        }
        """
        self.files = files
        self.data = {}  

    def load_country_data(self, country: str):
        """Load new cases (3rd column) for a given country."""
        path = self.files[country]
        new_cases = np.loadtxt(path, skiprows=1, usecols=[2])  # 3rd column
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

    def plot_country(self, country: str):
        """Plot new and cumulative cases for a given country."""
        days, new_cases, cum_cases = self.get_data(country)

        # New cases
        plt.figure()
        plt.plot(days, new_cases, label="New cases")
        plt.xlabel("Time [days]")
        plt.ylabel("New cases")
        plt.title(f"{country}: New cases")
        plt.legend()
        plt.show()

        # Cumulative cases
        plt.figure()
        plt.plot(days, cum_cases, label="Cumulative cases")
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative cases")
        plt.title(f"{country}: Cumulative cases")
        plt.legend()
        plt.show()

    def plot_all(self):
        """Plot all countries in the files dictionary."""
        for country in self.files.keys():
            self.plot_country(country)
