# eFusor: Extended Decision Fusion

[![PyPI - Version](https://img.shields.io/pypi/v/efusor)](https://pypi.org/project/efusor)
![PyPI - Status](https://img.shields.io/pypi/status/efusor)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/efusor)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/efusor)](https://pypistats.org/packages/efusor)

## Decision Fusion

__Decision Fusion__ is a combination of the decisions of multiple classifiers into a common decision;
i.e. a classifier ensemble operation.
The fusion of prediction vectors from multiple classifiers to a single prediction vector, 
out of which the decision is taken via `argmax`.

eFusor library provides an interface to common Decision Fusion methods;
such as [Majority Voting](https://en.wikipedia.org/wiki/Majority_rule) and 
less known [Tournament-style Borda Counting](https://en.wikipedia.org/wiki/Borda_count), 
as well as basic operations like `max` and `average`; 
implemented using [`numpy`](https://numpy.org).

The expected input for fusion is either `tensor` or `matrix`.  

- `Vector = list[float]` -- ordered list of predictions scores from a model for a query
- `Matrix = list[Vector]` -- ordered list of vectors; prediction scores for a query from a number of models
- `Tensor = list[Matrix]` -- ordered list of matrices; batch of predictions for several documents

## Motivation

[scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html) 
provides common [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) 
methods to combine the predictions of several classifiers;
and train meta predictors. 
An alternative to the ensemble learning methods is to use a heuristic.

eFusor provides this heuristic based decision fusion functionality.

eFusor was developed specifically to address the scenario 
where predictors (classifiers) may have different label spaces. 
Consequently, the library makes distinction between classes predicted with a low score (`0.0`)
and not predicted classes (`nan`).

## Vectorization

eFusor provides a `vectorize` function to do the vectorization 
making distinction between predicted and not predicted classes.
The function expects a `list` of class labels 
and a `dict` of prediction scores.

```python
from efusor import vectorize

labels = ["A", "B", "C", "D"]
scores = {"A": 0.75, "B": 0.25, "C": 0.00}

vector = vectorize(labels, scores)
# array([0.75, 0.25, 0.  ,  nan])
```

The function supports scores input as a vector, a matrix or a tensor.
That is a dict, a list of dicts or a list of lists of dicts.

## Fusion Methods

### Basic Fusion Methods

Since decision fusion of prediction vectors boils down to 
the reduction of a matrix to a vector column-wise, 
i.e. reducing a column vector to a scalar; 
any mathematical operation on a vector of numbers applies.

In Kittler, Hatef, Duin, and Matas (1998) "On Combining Classifiers". 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 20-3. 
The authors use the functions below as basic classifier combination schemes.


| method    | notes                                                                     |
|:----------|:--------------------------------------------------------------------------|
| `average` | mean value of a vector; requires well calibrated scores.                  |
| `product` | product rule and product rule issues!                                     |
| `sum`     | approximation of `product`; assumes posteriors to be not far from priors! |
| `max`     | approximation of `sum`                                                    |
| `min`     | bound version of `product`                                                |
| `median`  | approximation of `sum`; robust version of `average`                       |


### Voting Fusion Methods

The basic fusion methods operate with the classifier prediction scores, a real number vectors.
The problem could be reduced to operate on one-hot vectors;
in a way first taking per-classifier decision, rather than postponing it.
Combination of decision vectors is commonly done as a majority rule. 

`scikit-learn` provides [`VotingClassifier`](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) 
as an ensemble method and makes distinction between Hard Voting and Soft Voting.
While Hard Voting is the Majority Voting;
Soft Voting is nothing other than an average
(or weighted arithmetic mean, if weights are provided).

#### Rank-based Voting Methods

Rank-based voting, specifically [tournament-style borda count](https://en.wikipedia.org/wiki/Borda_count), 
is a decision technique commonly used is election decisions. 
While majority voting transforms prediction scores to a one-hot vector;
rank-based voting transforms it to an integer vector of ranks 
(the higher the score the lower the rank).

The benefit is that we still consider all predictions for fusion and 
do not require well calibrated scores.

### Weighted Fusion
In certain scenarios (e.g. fusion of decisions of rule-based and machine learning predictors),
it is desired to weigh different classifiers differently. 
[Weighted Average](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean) is a commonly used scheme.

`soft_voting` (an average) and `hard_voting` both implement weighted fusion.

(While Borda Count also allows to weigh different classifiers differently, it is not implemented).

#### Priority Fusion

An alternative to the weighted fusion is to select a prediction vector from a matrix
with respect to the weight vector.
However, in the scenario where predictors are allowed to have different label spaces,
this could lead to the final decision to be an all-NaN vector.

The `priority` fusion method implements such a heuristic,
and yielding the first non-NaN prediction vector from a matrix with respect to the weight vectors.
In case of equal weight values, a `max` fusion is applied on the set.


## Usage:

The primary decision fusion function is `fuse`. 

```python

from efusor import fuse

methods = [
    "max", "min", "sum", "product", "median", "average", 
    "hard_voting", "soft_voting", 
    "borda"
]

matrix = [[0.25, 0.60, 0.15], [0.00, 0.80, 0.00]]
weight = [0.75, 0.25] 

# unweighted results
for method in methods:
    result = fuse(matrix, method=method, digits=3)
    print(f"{method:<16}: {result}")
```

```text
max             : [0.25, 0.8, 0.15]
min             : [0.0, 0.6, 0.0]
sum             : [0.0, 1.067, 0.0]
product         : [0.0, 0.16, 0.0]
median          : [0.125, 0.7, 0.075]
average         : [0.125, 0.7, 0.075]
hard_voting     : [0, 2, 0]
soft_voting     : [0.125, 0.7, 0.075]
borda           : [1.0, 4.0, 0.0]
```

### Weighted Decision Fusion

```python
from efusor import fuse

matrix = [[0.25, 0.60, 0.15], [0.00, 0.80, 0.00]]
weight = [0.75, 0.25] 

for method in ["hard_voting", "soft_voting"]:
    result = fuse(matrix, method=method, digits=3, weights=weight)
    print(f"{method:<16}: {result}")
```

(rounded for readability)

```text
hard_voting     : [0.0, 1.0, 0.0]
soft_voting     : [0.188, 0.65, 0.112]
```


### Priority Decision Fusion

- requires `weights` (priorities)

```python
from efusor import fuse

matrix = [[0.25, 0.60, 0.15], [0.00, 0.80, 0.00]]
weight = [0.75, 0.25] 

for method in ["priority"]:
    result = fuse(matrix, method=method, digits=3, weights=weight)
    print(f"{method:<16}: {result}")
```

```text
priority        : [0.25, 0.6, 0.15]
```
