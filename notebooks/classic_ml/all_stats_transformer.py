import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import librosa, pandas as pd
from typing import Self, Sequence
from stat_feat_transformer import StatFeatTransformer
from numpy.typing import ArrayLike


def _extract_features(y, sr, n_mfcc):
    feats = {}
    feats["MFCC"] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).astype(np.float32)
    feats["Delta"] = librosa.feature.delta(feats["MFCC"])
    feats["DeltaDelta"] = librosa.feature.delta(feats["MFCC"], order=2)

    feats["SpectralCentroid"] = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats["SpectralBandwidth"] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats["SpectralRolloff"] = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats["SpectralFlatness"] = librosa.feature.spectral_flatness(y=y)
    feats["SpectralFlux"] = np.diff(feats["SpectralCentroid"], axis=1, prepend=0)

    # feats["F0"] = librosa.yin(y, fmin=50, fmax=300, sr=sr)[None, :]

    feats["RMS"] = librosa.feature.rms(y=y)
    feats["Energy"] = feats["RMS"] ** 2
    feats["ZCR"] = librosa.feature.zero_crossing_rate(y)

    # feats["Chroma"] = librosa.feature.chroma_stft(y=y, sr=sr)
    # feats["Tonnetz"] = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    return feats

class AllStatsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_mfcc: int = 100, n_jobs: int | None = -1):
        self.n_mfcc = n_mfcc
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike = None) -> Self:
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        X = X.copy()
        features_list = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_extract_features)(y, sr, self.n_mfcc) for y, sr in zip(X["Audio"], X["SR"])
        )

        all_keys = list(features_list[0].keys())
        for key in all_keys:
            X[key] = [f[key] for f in features_list]

        X = StatFeatTransformer(all_keys, n_frames=1).transform(X)
        X = X.drop(columns=["Audio", "SR"])
        return X.fillna(0)
