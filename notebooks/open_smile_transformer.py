import opensmile as sm
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

_smile = sm.Smile(
    feature_set=sm.FeatureSet.ComParE_2016,
    feature_level=sm.FeatureLevel.Functionals  # → 88 признаков
)

class OpenSmileTransformer(BaseEstimator, TransformerMixin):
    """ (Audio, SR)  →  матрица (n_samples, 88) """
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        def process(audio, sr):
            df = _smile.process_signal(audio, sr)
            return df.values.squeeze()

        feats = Parallel(n_jobs=self.n_jobs)(
            delayed(process)(audio, sr) for audio, sr in zip(X["Audio"], X["SR"])
        )
        X = np.vstack(feats).astype(np.float32)
        return X
