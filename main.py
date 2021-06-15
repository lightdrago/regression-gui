import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing

import itertools
import math
import random

from regressors import Regressor

class NonNumericEndogError(TypeError):
    pass

class App:
    def __init__(self):
        self.df = None
        self.X = None
        self.Y = None
        self.enabled_regressors = None
        self.last_results = dict()
        self.last_plots = dict()
    
    def load_data(self, filename, has_headers, top_lines=0):
        self.df = pd.read_csv(filename, header=0 if has_headers else None)
        self.df = self.df.dropna()
        self.df_preview = self.df[:1000]
        if top_lines > 0:
            self.df = self.df[:top_lines]
            self.df_preview = self.df_preview[:top_lines]

    def get_headers(self):
        return self.df.columns
    
    def set_X(self, column_indices):
        self.X = self.df.iloc[:, column_indices]
    
    def set_Y(self, column_index):
        self.Y = self.df.iloc[:, [column_index]]
        if not pd.api.types.is_numeric_dtype(self.Y.iloc[:, 0]):
            le = sklearn.preprocessing.LabelEncoder()
            self.Y = pd.DataFrame(le.fit_transform(self.Y), columns=[self.Y.columns])
    
    def get_regressors(self):
        return Regressor.available_models()
    
    def set_regressors(self, indices):
        self.enabled_regressors = list(
            value
            for (key, value)
            in enumerate(Regressor.available_models())
            if key in indices
        )
        return self.enabled_regressors
    
    def run(self):
        for r in self.enabled_regressors:
            r.run(self.X, self.Y)
            self.last_results[r.title] = r.get_stats(self.Y, self.X)
            print(r.get_stats(self.Y, self.X))
    

if __name__ == '__main__':
    print("Нужно запускать gui.py!")
