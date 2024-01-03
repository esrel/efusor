# eFusor: Extended Decision Fusion


## Usage:

```python

from efusor import fuse

methods = [
    "max", "min", "sum", "product", "median", "average", 
    "hard_voting", "soft_voting", 
    "borda"
]

matrix = [[0.25, 0.60, 0.15], [0.00, 0.80, 0.00]]

# unweighted results
for method in methods:
    result = fuse(matrix, method=method)
    print(f"{method}: {result}")
```

```text
max:         [0.25, 0.8, 0.15]
min:         [0.0, 0.6, 0.0]
sum:         [0.0, 1.07, 0.0]
product:     [0.0, 0.16, 0.0]
median:      [0.125, 0.7, 0.075]
average:     [0.125, 0.7, 0.075]
hard_voting: [0, 2, 0]
soft_voting: [0.125, 0.7, 0.075]
borda:       [1.0, 4.0, 0.0]
```
