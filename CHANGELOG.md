# ChangeLog

## 0.1.4

2024.01.12

- added `cutoff` threshold to `fuse`
- added `scaled` kwarg to `fuse` to apply `softmax` to fused vectors

## 0.1.3

2024.01.09

- added `softmax` with `nan` support to `utils.py`

## 0.1.2

2024.01.07

- added vectorization from dict of scores
- added `priority` fusion

## 0.1.1

2024.01.06

- added vector based selection and re-ranking


## 0.1.0

2024.01.04: initial release

- `basic`, `voter` and `borda` (simple and tournament-style) decision fusion methods.
- support function for vector scaling (`scale`).
- ranking functions (`rank`)