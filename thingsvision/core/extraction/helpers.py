import numpy as np

Array = np.ndarray
AxisError = np.AxisError


def center_features(X: Array) -> Array:
    """Center features to have zero mean."""
    X -= X.mean(axis=0)
    return X


def normalize_features(X: Array) -> Array:
    """Normalize feature vectors by their l2-norm."""
    try:
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
    except AxisError:
        raise Exception(
            "\nMake sure that features are represented as an n-dimensional NumPy array\n"
        )
    return X
