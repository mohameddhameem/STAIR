"""Feature transforms kept in an importable module for persisted sklearn models."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
import replay

class NumericTextStats(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None):return self
    def transform(self,X):return np.asarray([replay.numeric(t) for t in X],np.float32)
