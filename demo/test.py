""" eFusor Test (demo) """

from efusor import fuse


methods = [
    "max", "min", "sum", "product", "median", "average",
    "hard_voting", "soft_voting",
    "borda"
]

matrix = [[0.25, 0.60, 0.15], [0.00, 0.80, 0.00]]
weight = [0.75, 0.25]

print()
print("unweighted fusion:")
for method in methods:
    result = fuse(matrix, method=method, digits=3)
    print(f"{method:<16}: {result.index(max(result))} : {result}")

print()
print("weighted fusion:")
for method in ["hard_voting", "soft_voting"]:
    result = fuse(matrix, method=method, digits=3, weights=weight)
    print(f"{method:<16}: {result.index(max(result))} : {result}")

print()
print("priority fusion:")
for method in ["priority"]:
    result = fuse(matrix, method=method, digits=3, weights=weight)
    print(f"{method:<16}: {result.index(max(result))} : {result}")
