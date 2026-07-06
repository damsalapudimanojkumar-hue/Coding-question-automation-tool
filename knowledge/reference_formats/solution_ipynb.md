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