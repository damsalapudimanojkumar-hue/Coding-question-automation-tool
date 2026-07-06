# Detection Patterns

Find objects in student code by TYPE and STRUCTURE, never by variable name
(unless explicitly unavoidable and stated in the problem text).

## Pattern: detect a regression model
```python
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso

def find_model(student_code, model_classes):
    for v in student_code.values():
        if isinstance(v, model_classes) and hasattr(v, "coef_"):
            return v
    return None
```

## Pattern: detect a scaler and confirm it's fitted
```python
from sklearn.preprocessing import StandardScaler

def find_scaler(student_code, n_features):
    for v in student_code.values():
        if isinstance(v, StandardScaler) and hasattr(v, "mean_") and len(v.mean_) == n_features:
            return v
    return None
```

## Pattern: detect PolynomialFeatures with specific degree
```python
from sklearn.preprocessing import PolynomialFeatures

def find_poly(student_code, degree):
    for v in student_code.values():
        if isinstance(v, PolynomialFeatures) and v.degree == degree:
            return v
    return None
```

## Pattern: detect a fitted classifier
```python
def find_classifier(student_code, model_classes):
    for v in student_code.values():
        if isinstance(v, model_classes) and hasattr(v, "classes_"):
            return v
    return None
```

## Pattern: detect a PyTorch model
```python
import torch.nn as nn

def find_torch_model(student_code):
    for v in student_code.values():
        if isinstance(v, nn.Module) and any(p.requires_grad for p in v.parameters()):
            return v
    return None
```

## Pattern: detect predictions array with expected shape
```python
import numpy as np

def find_predictions(student_code, expected_len):
    for v in student_code.values():
        if isinstance(v, np.ndarray) and v.shape == (expected_len,):
            return v
    return None
```

## Pattern: detect a clustering model
```python
from sklearn.cluster import KMeans

def find_kmeans(student_code, n_clusters):
    for v in student_code.values():
        if isinstance(v, KMeans) and v.n_clusters == n_clusters:
            return v
    return None
```
For clustering ground-truth comparison, use the Hungarian algorithm to
align predicted cluster labels with true labels before computing agreement
(cluster label IDs are arbitrary, not consistent with ground truth IDs).

## Pattern: name-first with type fallback (only when unavoidable)
```python
def find_by_name_or_type(student_code, name, expected_type):
    if name in student_code and isinstance(student_code[name], expected_type):
        return student_code[name]
    for v in student_code.values():
        if isinstance(v, expected_type):
            return v
    return None
```
Use only for things like `word2idx`, `tokenizer` where structural detection
isn't reliable. Always tell students the exact expected variable name in
the problem statement when this pattern is used.
