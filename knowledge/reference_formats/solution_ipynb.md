# solution.ipynb — Reference Format

## Cell Structure (exactly 3 cells)

### Cell 1 — Markdown
Contains the short problem summary and task list.
Keep it brief — the full problem statement is in question.json.

### Cell 2 — Code (pre-filled, student does NOT edit)
```python
# Run this cell before writing your solution
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
# ... other imports relevant to the assignment

# Load data from S3
dataset_train = pd.read_csv('https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/<path>/dataset_train_<slug>.csv')
dataset_test = pd.read_csv('https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/<path>/dataset_test_<slug>.csv')

print('Train shape:', dataset_train.shape)
print('Test shape:', dataset_test.shape)
```

### Cell 3 — Code (student writes here — BLANK)
```python
# Write your code here

```

## Absolute Rules
- Cell 3 must contain ONLY `# Write your code here` — no solution code
- No `plt.show()` anywhere
- No `!pip install` anywhere  
- No `/content/` paths (Colab-only)
- No em dashes in markdown cell
- `warnings.filterwarnings('ignore')` always in cell 2
- Data loads from S3 URLs, not local CSV paths
- S3 URL pattern: `https://s3.ap-south-1.amazonaws.com/new-assets.ccbp.in/frontend/content/aiml/<subfolder>/<filename>.csv`

## Image / Computer-Vision variant (Cell 2)

For CV assignments the dataset is REAL IMAGE FILES in folders, never a CSV of
pixels. Cell 2 loads them with `torchvision.datasets.ImageFolder` from the local
training folder (present in the working directory), not from S3:

```python
# Run this cell before writing your solution
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Training images: one subfolder per class (folder name = class label)
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # match the size stated in the problem
    transforms.ToTensor(),         # scales pixels to [0, 1], channel-first (C, H, W)
])
train_dataset = datasets.ImageFolder('dataset_train_<CODE>', transform=transform)
class_names = train_dataset.classes  # exact class-label folder names

# Unlabeled test images the student must predict live here (flat folder):
TEST_DIR = 'dataset_test_<CODE>'
```

CV cell-2 notes:
- Point `ImageFolder` at the LOCAL `dataset_train_<CODE>` directory — no S3, no
  `urllib`, no downloads, no pixel CSV.
- The flat `dataset_test_<CODE>/` folder has NO labels; the true labels live in
  `tests/ground_truth_<CODE>.csv` (`filename,label`) and are used only by the grader.
- Use the exact image size / transforms the problem statement specifies so the
  student's preprocessing matches how the hidden test images are scored.