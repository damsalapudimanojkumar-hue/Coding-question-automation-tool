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

## Pattern: detect a CV classifier by BEHAVIOR (image assignments)
Do not detect image models by variable name. Filter every `nn.Module` through a
forward-pass probe on the exact input shape and required class count. Skip
`__`-prefixed internal keys.
```python
import torch
import torch.nn as nn

def outputs_n_classes(module, in_shape=(1, 3, 32, 32), n_classes=10):
    """Behavioral probe: accepts (N, C, H, W) and returns n_classes logits."""
    try:
        module.eval()
        with torch.no_grad():
            out = module(torch.zeros(*in_shape))
        return hasattr(out, "shape") and out.shape[-1] == n_classes
    except Exception:
        return False

def find_cv_model(student_code, in_shape=(1, 3, 32, 32), n_classes=10):
    for name, v in student_code.items():
        if name.startswith("__"):
            continue
        if isinstance(v, nn.Module) and outputs_n_classes(v, in_shape, n_classes):
            return v
    return None
```
When several models exist (e.g. LeNet vs improved vs Inception), disambiguate by
architecture signature, not name: collect `Conv2d` kernel sizes and counts.
```python
def conv_kernels(module):
    return [tuple(m.kernel_size) if isinstance(m.kernel_size, (tuple, list))
            else (m.kernel_size, m.kernel_size)
            for m in module.modules() if isinstance(m, nn.Conv2d)]

def count_1x1(module):
    return sum(1 for k in conv_kernels(module) if k == (1, 1))
# e.g. an Inception module: distinct kernel sizes >= 3 and count_1x1(m) >= 3;
# a dropout upgrade: any(isinstance(m, nn.Dropout) for m in module.modules()).
```

## Pattern: score a CV model against an image ground-truth manifest
```python
import os, torch
import pandas as pd
from PIL import Image
from torchvision import transforms

def cv_accuracy(model, test_dir, gt_csv, tfm, batch=100):
    gt = pd.read_csv(gt_csv)  # columns: filename,label
    X = torch.stack([
        tfm(Image.open(os.path.join(test_dir, f)).convert("RGB"))
        for f in gt["filename"]
    ])
    y = torch.tensor(list(gt["label"]))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            preds.append(model(X[i:i + batch]).argmax(dim=1))
    return (torch.cat(preds) == y).float().mean().item()
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
