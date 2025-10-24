from sklearn.base import BaseEstimator, TransformerMixin

class TextFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf):
        self.tfidf = tfidf
    def fit(self, X, y=None):
        self.tfidf.fit(X["__joined_text__"].fillna(""))
        return self
    def transform(self, X):
        return self.tfidf.transform(X["__joined_text__"].fillna(""))

class DenseCT(BaseEstimator, TransformerMixin):
    def __init__(self, ct):
        self.ct = ct
    def fit(self, X, y=None):
        self.ct.fit(X, y)
        return self
    def transform(self, X):
        return self.ct.transform(X)