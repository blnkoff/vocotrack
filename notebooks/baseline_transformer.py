import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import librosa, pandas as pd
from typing import Self, Sequence
from stat_feat_transformer import StatFeatTransformer
from numpy.typing import ArrayLike


# ---------- быстрый Baseline ----------
def _mfcc_single(y, sr, n_mfcc):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).astype(np.float32)

class BaselineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_mfcc: int = 100, n_jobs: int | None = -1):
        self.n_mfcc = n_mfcc
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike = None) -> Self:
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        X = X.copy()
        # Параллельно MFCC
        mfccs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_mfcc_single)(y, sr, self.n_mfcc) for y, sr in zip(X["Audio"], X["SR"])
        )
        X["MFCC"] = mfccs

        X = StatFeatTransformer(["MFCC"]).transform(X)
        return X.drop(columns=["Audio", "SR"])
