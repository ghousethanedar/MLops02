## import the requried libraries
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


## Drop not necessary columns

class Dropcols(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X


# ## Missing Flag

# class Missing(BaseEstimator,TransformerMixin):
#     def __init__(self,variables=None):
#         pass

#     def fit(self,X,y=None):
#         pass

#     def transform(self,X,y=None):
#         pass


# ## imputing the missing categorical variables

class Category_imputer(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.cat_impute = {}
        for feature in self.variables:
            self.cat_impute[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.cat_impute[feature])
        # print(X.head())
        return X


## imputing the numerical variables

class Numerical_imputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.num_encoder = {}
        for feature in self.variables:
            self.num_encoder[feature] = X[feature].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.num_encoder[feature])
        # print(X.head())
        return X


## Rare variables

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    # persist frequent labels in dictionary
    def fit(self, X, y=None):
        self.rare_encoding = {}
        X = X.copy()
        for feature in self.variables:
            t = pd.Series(X[feature].value_counts() / np.float(len(X)))
            self.rare_encoding[feature] = list(t[t >= self.tol].index)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.rare_encoding[feature]), X[feature], 'Rare')
        # print(X.head)
        return X


## One hot encoding the categorical variables

class Categorical_encoding(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        X = X.copy()
        self.dummies = pd.get_dummies(X, drop_first=True).columns
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = pd.concat([X,
                       pd.get_dummies(X[self.variables], drop_first=True)],
                      axis=1)

        X.drop(labels=self.variables, axis=1, inplace=True)

        # add missing dummies if any
        missing_vars = [var for var in self.dummies if var not in X.columns]

        if len(missing_vars) != 0:
            for var in missing_vars:
                X[var] = 0
        # print(X.head())
        return X







