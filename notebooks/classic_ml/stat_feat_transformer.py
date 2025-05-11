from typing import Sequence, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, kurtosis

def _statistics(batch_3d: np.ndarray, q: float = 0.25) -> np.ndarray:
    """
    batch_3d: shape (B, n_rows, n_cols)
    Возвращает массив shape (B, n_rows, 7) со статистиками
    """
    mean  = batch_3d.mean(axis=2)
    var   = batch_3d.var(axis=2)
    skew_ = skew(batch_3d, axis=2, nan_policy="omit")
    kurt  = kurtosis(batch_3d, axis=2, nan_policy="omit")
    min_  = batch_3d.min(axis=2)
    max_  = batch_3d.max(axis=2)
    q25   = np.quantile(batch_3d, q, axis=2)
    return np.stack((mean, var, skew_, kurt, min_, max_, q25), axis=2)

class StatFeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features: Sequence[str | int], n_frames: int = 1):
        self.features = features
        self.n_frames = n_frames

    def fit(self, X: ArrayLike, y: ArrayLike = None) -> Self:
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        X = X.copy()
        out_parts = []

        for feat in self.features:
            arrs = X[feat].to_list()
            max_len = max(a.shape[1] for a in arrs)
            n_mfcc = arrs[0].shape[0]
            frame_len = max_len // self.n_frames

            batch = np.zeros((len(arrs), n_mfcc, max_len), dtype=np.float32)
            for i, a in enumerate(arrs):
                batch[i, :, :a.shape[1]] = a

            frame_stats_list = []
            for f in range(self.n_frames):
                start = f * frame_len
                end = (f + 1) * frame_len if f < self.n_frames - 1 else max_len
                frame = batch[:, :, start:end]
                stats = _statistics(frame)  # shape=(B, n_mfcc, 7)
                frame_stats_list.append(stats)

            all_stats = np.concatenate(frame_stats_list, axis=2)  # along stat axis
            B = all_stats.shape[0]
            out = all_stats.reshape(B, -1)

            stat_names = ('mean', 'var', 'skew', 'kurt', 'min', 'max', 'q25')
            cols = [f"{feat}_f{f}_{s}_{i}" for f in range(self.n_frames)
                                              for i in range(n_mfcc)
                                              for s in stat_names]
            out_parts.append(pd.DataFrame(out, columns=cols))

        X = X.drop(columns=self.features, axis=1).reset_index(drop=True)
        X: pd.DataFrame = pd.concat([X] + out_parts, axis=1)

        return X