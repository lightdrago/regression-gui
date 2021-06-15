from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn import tree, svm, gaussian_process
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import sklearn.metrics
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
import uuid
from datetime import datetime
from pathlib import Path

class PolynomialRegressionWrapper():
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None
    
    def fit(self, *args, **kwargs):
        self.model = Pipeline([('poly', PolynomialFeatures(degree=self.degree)),
                               ('linear', LinearRegression(fit_intercept=False))])
        self.model = self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

class LinearRegressionWrapper():
    def __init__(self, column_config=None):
        self.model = None
        self.result = None
        if column_config:
            self.column_config = column_config
            self.column_morph = lambda x: tuple(c[1](x) for c in column_config)
        else:
            self.column_config = self.column_morph = None
    
    def fit(self, X, Y, *args, **kwargs):
        if self.column_morph:
            frames = []
            for column in X.columns:
                frames.append(
                    np.column_stack(
                        self.column_morph(X[column].values)
                    )
                )
            X_ = np.column_stack(frames)
        else:
            X_ = X.values
        
        N = X_.shape[0]
        p = X_.shape[1] + 1
        
        X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p] = X_

        self.X_with_intercept = X_with_intercept
        self.model = sm.OLS(Y, X_with_intercept)
    
    def predict(self, *args, **kwargs):
        return self.model.fit()


class Regressor():
    def __init__(self, cls, title, *args, is_sklearn=True, **kwargs):
        self.cls = cls
        self.title = title
        self.is_sklearn = is_sklearn
        self.args = args
        self.kwargs = kwargs
        self.last_instance = None
        self.last_prediction = None
        self.last_plot = None
    
    def run(self, X, Y):
        reg = self.last_instance = self.cls(*(self.args), **(self.kwargs))
        print(f'Fitting...')
        reg.fit(X, Y)
        print(f'Predicting...')
        self.last_prediction = reg.predict(X)
        return self.last_prediction
    
    def norm_r2_score(self, Y, Y_pred, X):
        n = len(X)
        p = X.shape[1]
        r2 = sklearn.metrics.r2_score(Y, Y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def str_error(self, Y, Y_pred, X):
        n = len(X)
        p = X.shape[1]
        tss = sklearn.metrics.mean_squared_error(Y, Y_pred) * (n / (n - p - 1))
        return math.sqrt(tss)

    
    def get_stats(self, Y, X):
        if self.is_sklearn:
            return self.evaluate(Y, X, self.last_prediction)
        else:
            # Y_pred = self.last_prediction.predict(self.last_instance.X_with_intercept)
            return self.extract_summary(Y, X)
    
    def extract_summary(self, Y, X):
        summary = self.last_prediction.summary()

        # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html
        data = self.last_prediction

        std_err = math.sqrt(data.centered_tss / data.df_resid)

        template = "{0:>50}: {1:8.4f}\n"
        head = (template.format('R-квадрат', data.rsquared) + 
               template.format('Нормированный R-квадрат', data.rsquared_adj) + 
               template.format('Критерий Фишера (F)', data.fvalue) + 
               template.format('p(F)', data.f_pvalue) + 
               template.format('Стандартная ошибка', std_err)
        )

        table = summary.tables[1].data
        body = ""

        template = "   {0:<6} │ {1:>10} │ {2:>10} │ {3:>10} │ {4:>10} │ {5:>10} │ {6:>10} \n"
        body += template.format(' ', 'Коэф.', 'Стд. ош.', 't', 'P>|t|', '[ 0.025  ', ' 0.975 ] ')
        body += ("  " + "─" * 8 + "┼" + 
         "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + 
         "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "\n")

        morph_cfg = self.last_instance.column_config
        if morph_cfg:
            column_names = itertools.chain(iter(('const',)), 
                iter(
                    morph.format(f'x{original}')
                    for original, morph
                    in itertools.product(range(1, X.shape[1]+1), (morph_column[0] for morph_column in morph_cfg))
                )
            )

            for row in table[1:]:
                body += template.format(
                    next(column_names),
                    *[cell.strip() for cell in row[1:]]
                )
        else:
            for row in table[1:]:
                body += template.format(*[cell.strip() for cell in row])

        return head + "\n\n" + body
    
    def save_summary(self, Y, X):
        summary = self.get_stats(Y, X)
        folder = Path("Отчёты")
        folder.mkdir(exist_ok=True)
        filename = folder / Path(f"{datetime.now().strftime('%x %X')} - {self.title}.txt".replace(':', '_').replace('^', '').replace('/', '_'))
        with open(str(filename), 'wb') as f:
            f.write(summary.encode('utf-8'))
    
    def save_plot_image(self, Y, X):
        plt.ioff()
        fig, ax = plt.subplots()
        ax.scatter(X, Y, color='blue')
        if self.is_sklearn:
            ax.plot(X, self.last_prediction, color='red')
        else:
            params = self.last_prediction.params
            x = np.linspace(X.values.min(), X.values.max())
            if self.last_instance.column_morph:
                raw_morphs = self.last_instance.column_morph(x)
                weighted_morphs = [morph * weight for morph, weight in zip(raw_morphs, params.values[1:])]
                const = np.repeat(params.const, len(x))
                y = np.column_stack([const] + weighted_morphs).sum(axis=1)
            else:
                y = params.x1 * x + params.const
            ax.plot(x, y, color='red')
        folder = Path("Отчёты")
        folder.mkdir(exist_ok=True)
        filename = folder / Path(f"{datetime.now().strftime('%x %X')} - {self.title}.png".replace(':', '_').replace('^', '').replace('/', '_'))
        # filename = str(uuid.uuid4().hex) + '.png'
        fig.savefig(str(filename))
        return str(filename)


    def evaluate(self, Y, X, Y_pred):
        Y_pred = self.last_prediction
        r2_score = sklearn.metrics.r2_score(Y, Y_pred)
        norm_r2_score = self.norm_r2_score(Y, Y_pred, X)
        MAE = sklearn.metrics.mean_absolute_error(Y, Y_pred)
        MSE = sklearn.metrics.mean_squared_error(Y, Y_pred)
        std_error = self.str_error(Y, Y_pred, X)
        max_error = sklearn.metrics.max_error(Y, Y_pred)
        size = Y._data.shape[1]

        scores = [
            ('R-квадрат', r2_score),
            ('Нормированный R-квадрат', norm_r2_score),
            ('Среднеквадратическая ошибка (MSE)', MSE),
            ('Средняя абсолютная ошибка (MAE)', MAE),
            ('Стандартная ошибка', std_error),
            ('Максимальная ошибка', max_error),
        ]
        template = "{0:>50}: {1:8.4f}"
        return '\n'.join(template.format(*score) for score in scores)

    @classmethod
    def available_models(cls):
        return [
            cls(LinearRegressionWrapper, 'Линейная регрессия', is_sklearn=False),
            cls(tree.DecisionTreeRegressor, 'CART-регрессия'),
            cls(svm.SVR, 'SVM-регрессия'),
            cls(gaussian_process.GaussianProcessRegressor, 'Гауссовская регрессия'),
            cls(Ridge, 'Ridge-регрессия'),
            cls(
                LinearRegressionWrapper, 
                'Линейная регрессия (x^2)', is_sklearn=False, 
                # column_morph=lambda x: (x, np.square(x))),
                column_config=(
                    ('{0}', lambda x: x),
                    ('{0}^2', lambda x: np.square(x)),
                )
            ),
            cls(
                LinearRegressionWrapper, 
                'Линейная регрессия (x^3)', is_sklearn=False, 
                # column_morph=lambda x: (x, np.square(x), np.power(x, 3))
                column_config=(
                    ('{0}', lambda x: x),
                    ('{0}^2', lambda x: np.square(x)),
                    ('{0}^3', lambda x: np.power(x, 3)),
                )
            ),
        ]